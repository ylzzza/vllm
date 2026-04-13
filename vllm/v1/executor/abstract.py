# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from functools import cached_property
from typing import TYPE_CHECKING, Literal, TypeVar, overload

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase

logger = init_logger(__name__)

_R = TypeVar("_R")

FailureCallback = Callable[[], None]


class Executor(ABC):
    """vLLM 执行器的抽象基类。

    `Executor` 代表“调度器之下、worker 之上”的执行层抽象。
    它的职责不是决定“这一轮跑哪些请求”，而是把 scheduler 产出的
    `SchedulerOutput` 分发给底层 worker 去真正执行。

    它既可以对应单设备执行器，也可以对应多设备/多进程分布式执行器。
    上层 `EngineCore` 只依赖这层统一接口，而不需要关心底层到底是：

    1. 单进程本地执行
    2. 多进程执行
    3. Ray 分布式执行
    4. 外部 launcher 管理的执行
    """

    uses_ray: bool = False  # 是否使用 Ray 进行编排。
    supports_pp: bool = False  # 是否支持 pipeline parallel。

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        executor_class: type[Executor]
        parallel_config = vllm_config.parallel_config
        distributed_executor_backend = parallel_config.distributed_executor_backend
        # distributed_executor_backend 会在 VllmConfig.__post_init__ 中补齐。
        # 这里根据配置选择真正的执行器实现类。
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "ray":
            from vllm.v1.executor.ray_executor import RayDistributedExecutor

            executor_class = RayDistributedExecutor
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor

            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "uni":
            from vllm.v1.executor.uniproc_executor import UniProcExecutor

            executor_class = UniProcExecutor
        elif distributed_executor_backend == "external_launcher":
            # TODO: 需要先让 v1 scheduling 的行为完全确定化，
            # 才能更好支持 external launcher 场景。
            executor_class = ExecutorWithExternalLauncher
        elif isinstance(distributed_executor_backend, str):
            executor_class = resolve_obj_by_qualname(distributed_executor_backend)
            if not issubclass(executor_class, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {executor_class}."
                )
        else:
            raise ValueError(
                f"Unknown distributed executor backend: {distributed_executor_backend}"
            )
        return executor_class

    @instrument(span_name="Executor init")
    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        # 复制常用配置字段，便于子类和执行路径直接访问。
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self._init_executor()
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()
        # 当存在 KV connector 且需要跨 worker 聚合其输出时使用。
        self.kv_output_aggregator: KVOutputAggregator | None = None

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """
        根据配置初始化底层 worker 的 KV cache，并完成执行前预热。

        运行主线如下：
        1. 通过 RPC 把 `kv_cache_configs` 下发给所有 worker
        2. 让每个 worker 分配并初始化自己的 KV cache
        3. 触发 worker 侧编译/预热
        4. 把 worker 侧统计到的编译时间回写到主进程配置中
        """
        self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))
        compilation_times: list[float] = self.collective_rpc("compile_or_warm_up_model")
        # 多卡场景下，真正的编译通常发生在 worker 进程内，主进程自己的配置对象
        # 并不会自动更新。因此这里把 worker 侧的编译耗时聚合回主进程。
        # 由于各 worker 是并行编译的，这里取最大值更符合整体初始化耗时。
        if compilation_times:
            self.vllm_config.compilation_config.compilation_time = max(
                compilation_times
            )

    def register_failure_callback(self, callback: FailureCallback):  # noqa: B027
        """
        注册一个失败回调。

        当 executor 进入不可恢复的失败状态时，会调用这个函数通知上层。
        默认实现为空，具体行为由子类决定。
        """
        pass

    def determine_available_memory(self) -> list[int]:  # 单位：bytes
        return self.collective_rpc("determine_available_memory")

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        return self.collective_rpc("get_kv_cache_spec")

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[False] = False,
    ) -> list[_R]:
        """
        在所有 worker 上执行一次 RPC 调用。

        参数：
            method: 要执行的 worker 方法名，或者一个会被序列化后发送给
                所有 worker 执行的可调用对象。

                如果 `method` 是可调用对象，它除了接收 `args/kwargs` 中的
                参数之外，还应额外接收一个 `self` 参数，这个 `self`
                就是目标 worker 对象。
            timeout: 最长等待秒数。超时会抛出 `TimeoutError`。
                `None` 表示一直等待。
            args: 传给 worker 方法的位置参数。
            kwargs: 传给 worker 方法的关键字参数。
            non_block: 若为 `True`，则立即返回 Future，而不是阻塞等待结果。

        返回：
            一个列表，包含每个 worker 的返回结果。

        说明：
            这个接口更适合传输控制面消息；真正的大数据面传输，通常应由
            专门的数据通道负责。
        """
        pass

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[True] = True,
    ) -> Future[list[_R]]:
        pass

    @abstractmethod
    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block: bool = False
    ):
        raise NotImplementedError

    def get_kv_connector_handshake_metadata(
        self,
    ) -> list[dict[int, KVConnectorHandshakeMetadata]]:
        return self.collective_rpc("get_kv_connector_handshake_metadata")

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[False] = False
    ) -> ModelRunnerOutput | None:
        pass

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput | None]:
        pass

    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # 默认实现直接把本轮 SchedulerOutput 广播到所有 worker。
        # 对于单设备执行器，等价于直接调用单个 worker；
        # 对于分布式执行器，子类可通过重写 collective_rpc / execute_model
        # 来优化“只收一个 rank 的返回结果”等细节。
        output = self.collective_rpc(  # type: ignore[call-overload]
            "execute_model", args=(scheduler_output,), non_block=non_block
        )
        return output[0]

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[False] = False
    ) -> ModelRunnerOutput:
        pass

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput]:
        pass

    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        # 当 execute_model 只完成前向而未返回采样结果时，再通过这个接口
        # 补做采样。
        output = self.collective_rpc(  # type: ignore[call-overload]
            "sample_tokens", args=(grammar_output,), non_block=non_block
        )
        return output[0]

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch")

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # draft token 目前通常只需从单个输出 worker 回收即可；
        # 默认实现仍沿用 collective_rpc，再取第一个结果。
        output: list[DraftTokenIds] = self.collective_rpc("take_draft_token_ids")
        return output[0]

    @property
    def max_concurrent_batches(self) -> int:
        return 1

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        self.collective_rpc("profile", args=(is_start, profile_prefix))

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.collective_rpc(
            "save_sharded_state",
            kwargs=dict(path=path, pattern=pattern, max_size=max_size),
        )

    @abstractmethod
    def check_health(self) -> None:
        """检查 executor 是否健康；若不健康，应抛出异常。"""
        raise NotImplementedError

    def shutdown(self) -> None:
        """关闭 executor。"""
        self.collective_rpc("shutdown")

    def init_kv_output_aggregator(self, connector: "KVConnectorBase") -> None:
        """初始化 KV 输出聚合器。"""
        self.kv_output_aggregator = KVOutputAggregator.from_connector(
            connector, self.parallel_config.world_size
        )

    @cached_property  # 避免重复执行不必要的 RPC。
    def supported_tasks(self) -> tuple[SupportedTask, ...]:
        output: list[tuple[SupportedTask, ...]]
        output = self.collective_rpc("get_supported_tasks")
        return output[0]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("add_lora", args=(lora_request,)))

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("remove_lora", args=(lora_id,)))

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("pin_lora", args=(lora_id,)))

    def list_loras(self) -> set[int]:
        sets: list[set[int]] = self.collective_rpc("list_loras")
        for s in sets:
            assert s == sets[0], "All workers should have the same LORAs."
        return sets[0]

    def reset_mm_cache(self) -> None:
        """重置每个 worker 中的多模态缓存。"""
        self.collective_rpc("reset_mm_cache")

    def reset_encoder_cache(self) -> None:
        """重置每个 worker 中的 encoder cache，清除已缓存的 encoder 输出。"""
        self.collective_rpc("reset_encoder_cache")

    def sleep(self, level: int = 1):
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return
        # 通过 RPC 让所有 worker 进入睡眠态，并记录仍处于睡眠中的资源标签。
        time_before_sleep = time.perf_counter()
        self.collective_rpc("sleep", kwargs=dict(level=level))
        time_after_sleep = time.perf_counter()
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True
        logger.info(
            "It took %.6f seconds to fall asleep.", time_after_sleep - time_before_sleep
        )

    def wake_up(self, tags: list[str] | None = None):
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return
        # tags=None 表示唤醒全部资源；否则仅唤醒指定标签。
        if tags:
            for tag in tags:
                if tag not in self.sleeping_tags:
                    logger.warning(
                        "Tag %s is not in sleeping tags %s", tag, self.sleeping_tags
                    )
                    return
        time_before_wakeup = time.perf_counter()
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        time_after_wakeup = time.perf_counter()
        logger.info(
            "It took %.6f seconds to wake up tags %s.",
            time_after_wakeup - time_before_wakeup,
            tags if tags is not None else self.sleeping_tags,
        )
        if tags:
            for tag in tags:
                self.sleeping_tags.remove(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        raise NotImplementedError


from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    ExecutorWithExternalLauncher as _ExecutorWithExternalLauncher,
)
from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    UniProcExecutor as _UniProcExecutor,
)

# 为了向后兼容，保留旧名字导出。
UniProcExecutor = _UniProcExecutor
ExecutorWithExternalLauncher = _ExecutorWithExternalLauncher
