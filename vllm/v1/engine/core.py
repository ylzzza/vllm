# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import queue
import signal
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from functools import partial
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, TypeVar, cast

import msgspec
import zmq

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.tracing import instrument, maybe_init_worker_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,
    EEPNotificationType,
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    PauseMode,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    get_device_indices,
)
from vllm.v1.executor import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import compute_iteration_details
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # collective_rpc 的返回类型

# 这个文件实现的是 vLLM V1 的“内核执行层”。
# 可以把它分成三层来看：
# 1. `EngineCore`：真正的调度与执行主循环，负责 scheduler + executor。
# 2. `EngineCoreProc`：把 `EngineCore` 包成后台进程，通过 ZMQ 和前端通信。
# 3. `DPEngineCoreProc`：在 `EngineCoreProc` 之上补齐 DP/MoE 场景下的 wave 协同、
#    负载统计和 elastic scaling 逻辑。


class EngineCore:
    # `EngineCore` 是 V1 引擎真正“干活”的地方。
    # 上层 client 只负责把请求送进来、把输出取回去；
    # 这里负责：
    # 1. 初始化模型执行器与 KV cache
    # 2. 构造 scheduler
    # 3. 每轮 schedule -> execute -> update_from_output
    # 4. 管理缓存、LoRA、sleep/wake、structured output 等运行时能力
    """vLLM 引擎的核心执行循环。"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        # 插件也需要在 engine/scheduler 这一层完成加载
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        self.vllm_config = vllm_config
        if not vllm_config.parallel_config.data_parallel_rank_local:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,
                vllm_config,
            )

        self.log_stats = log_stats

        # 先创建真正负责模型前向执行的 executor。
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self._eep_scale_up_before_kv_init()

        # 基于模型 profile 结果初始化 KV cache，并把最终 block 数同步回配置。
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
            vllm_config
        )

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        self.collective_rpc("initialize_cache", args=(num_gpu_blocks, num_cpu_blocks))

        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # scheduler 负责请求队列、batch 组装、block 分配、状态推进等逻辑。
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # 没有 KV cache 的 Encoder 模型不支持
            # chunked prefill。SSM 模型是否支持还有待确认。
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                vllm_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size = (
            vllm_config.cache_config.block_size
            * vllm_config.parallel_config.decode_context_parallel_size
            * vllm_config.parallel_config.prefill_context_parallel_size
        )

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

        self.mm_registry = mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(
            vllm_config
        )

        # 如果 scheduler 初始化了 KV connector，就需要从所有 worker
        # 收集握手元数据，这样 scheduler 侧的 connector 才能拿到完整上下文
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # 从 worker 收集并保存 KV connector 的传输握手元数据
            # （在 KV cache 注册完成之后）
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata 是来自 worker 的 dict 列表
                # 每个 dict 的结构都已经是 {tp_rank: metadata}
                # 这里把所有 worker 的 dict 合并成一个总 dict
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)

        # pipeline parallel 场景下，允许“调度下一批”和“等待上一批执行结果”
        # 交叠进行，减少 pipeline bubble。
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        self.is_ec_producer = (
            vllm_config.ec_transfer_config is not None
            and vllm_config.ec_transfer_config.is_ec_producer
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                scheduler_block_size, caching_hash_fn
            )

        # 根据是否启用 batch queue，选择主循环里调用的 step 实现。
        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        # socket 线程收到 abort 后会优先放入这里，主循环会在一次 forward
        # 结束后统一批量处理。
        self.aborts_queue = queue.Queue[list[str]]()

        self._idle_state_callbacks: list[Callable] = []

        # 把启动阶段分配的堆对象标记为静态，避免被 GC 扫描。
        # 这样可以减少老年代回收时的停顿。
        freeze_gc_heap()
        # 如果开启了 GC 调试，就在静态对象冻结后挂上调试回调。
        maybe_attach_gc_debug_callback()
        # 启用环境变量缓存（例如认为从这里开始不会再覆盖环境变量）。
        enable_envs_cache()

    @instrument(span_name="Prepare model")
    def _initialize_kv_caches(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int, KVCacheConfig]:
        start = time.time()

        # 先让 executor 告诉我们模型到底需要哪些 KV cache 规格。
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                # NOTE(yongji): 这里的值理论上应该已经在
                # _eep_scale_up_before_kv_init 阶段设置好了
                assert self.available_gpu_memory_for_kv_cache > 0
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # 通过 profile 得到“模型本体最多吃多少显存”，剩余部分才能分给 KV cache。
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # 无 attention 的模型不需要为 KV cache 预留显存
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)

        # 生成 KV cache 配置时，auto-fit 逻辑可能会把 max_model_len 调小，
        # 这里要把变化同步回所有 worker。
        max_model_len_before = vllm_config.model_config.max_model_len

        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )

        # 如果 auto-fit 把 max_model_len 调小了，要把新值同步给 worker。
        # 因为 worker 在显存 profile 之前就已经启动，缓存的仍然是原来的
        # （更大的）max_model_len。
        max_model_len_after = vllm_config.model_config.max_model_len
        if max_model_len_after != max_model_len_before:
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        num_cpu_blocks = 0

        # 真正创建 cache，并顺带完成一轮 warmup。
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info_once(
            "init engine (profile, create kv cache, warmup model) took %.2f seconds",
            elapsed,
            scope="local",
        )
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

    def add_request(self, request: Request, request_wave: int = 0):
        """把请求加入 scheduler。

        `request_wave` 表示该请求在 DP 场景下预期属于哪一轮 wave。
        """
        # 校验 request_id 的类型。
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        # pooling 请求不是“正常生成少走几步”，而是单独的任务类型，
        # 这里会先校验当前模型是否支持该 pooling task。
        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        # 从这里开始，请求正式交给 scheduler 持有和调度。
        self.scheduler.add_request(request)

    def abort_requests(self, request_ids: list[str]):
        """从 scheduler 中终止请求。"""

        # TODO: scheduler 实际上不一定需要知道具体的结束原因，
        # 后续再决定是否向下传播这类信息
        # （例如 client 主动 abort 还是命中了 stop 条件）。
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """执行模型；若失败则记录详细信息。"""
        try:
            yield
        except Exception as err:
            # 这里不捕获 BaseException，因为我们只关心 execute_model
            # 自身抛出的异常，并在这种情况下补充上下文信息。

            # NOTE: 这个方法自身保证不会再抛异常
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    @contextmanager
    def log_iteration_details(self, scheduler_output: SchedulerOutput):
        if not self.vllm_config.observability_config.enable_logging_iteration_details:
            yield
            return
        self._iteration_index = getattr(self, "_iteration_index", 0)
        iteration_details = compute_iteration_details(scheduler_output)
        before = time.monotonic()
        yield
        logger.info(
            "".join(
                [
                    "Iteration(",
                    str(self._iteration_index),
                    "): ",
                    str(iteration_details.num_ctx_requests),
                    " context requests, ",
                    str(iteration_details.num_ctx_tokens),
                    " context tokens, ",
                    str(iteration_details.num_generation_requests),
                    " generation requests, ",
                    str(iteration_details.num_generation_tokens),
                    " generation tokens, iteration elapsed time: ",
                    format((time.monotonic() - before) * 1000, ".2f"),
                    " ms",
                ]
            )
        )
        self._iteration_index += 1

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """完成一次调度、执行并生成输出。

        返回值是 `(outputs, model_executed)`，其中后者表示本轮是否真的执行了模型。
        """

        # 检查 scheduler 中是否还有请求，包括未完成的请求，
        # 以及已完成但尚未从 batch 中移除的请求。
        # 没有请求就不做任何事情。
        if not self.scheduler.has_requests():
            return {}, False
        # 1. 先让 scheduler 决定这一轮跑哪些 request / token。
        scheduler_output = self.scheduler.schedule()
        # 2. 交给 executor 执行模型前向。
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                # 某些路径下 execute_model 只做前向，采样要在这里补上。
                model_output = self.model_executor.sample_tokens(grammar_output)

        # 3. 先处理本轮执行期间收到的 abort，再推进 scheduler 状态。
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def post_step(self, model_executed: bool) -> None:
        # 使用异步调度时，无法提前拿到 draft token ids，
        # 因此这部分更新会在 worker 进程内完成，这里无需重复处理。
        if not self.async_scheduling and self.use_spec_decode and model_executed:
            # 取出 draft token ids。
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """在 batch queue 模式下完成调度和执行。

        注意：如果这一轮没有任何输出，会返回 `None`。

        执行流程如下：
        1. 如果 batch queue 没满，尝试继续调度新 batch。
           一旦成功调度，就直接返回空的 engine core 输出。
           也就是说，优先把 batch queue 填满，而不是优先取模型输出。
        2. 如果没有新 batch 可调度，说明 batch queue 已满，或者当前没有更多请求
           可以调度，此时阻塞等待队列中最早那个 batch 完成。
        3. 用执行结果更新 scheduler。
        """

        batch_queue = self.batch_queue
        assert batch_queue is not None

        # 这一版 step 主要给 pipeline parallel 用：
        # 队列没满时优先继续发批次，队列满了或没法继续调度时再等结果。
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            if not self.is_ec_producer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # pooling 没有 sampler；或者本轮其实没有真正执行模型，
                # 那就直接等 execute_model 的结果。
                # 不需要采样（本轮没有真正调度任何请求）。
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    # 当前不需要等待额外 token，可以直接拿 grammar 输出并立刻采样。
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # 要等上一轮的模型输出处理完成后，才能延迟执行这次采样。
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # 把这一轮对应的 future 放入队列。
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # 只要还能继续把队列填满，就尽量不要阻塞等结果，
                    # 这样吞吐更高。
                    # 只要队列还没满、也还有请求可调度，就不要阻塞等待
                    # 下一个 worker 的响应。
                    return None, True

        elif not batch_queue:
            # 队列为空。理论上不应该走到这里，因为只有在 scheduler 仍有请求
            # 或者队列非空时，才会调用这个方法。
            return None, False

        # 队列不能继续填了，就阻塞拿最早那个 batch 的结果。
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                # sample_tokens() 返回 None 表示原始 execute_model()
                # 调用失败，这里把那个异常重新抛出来。
                exec_model_fut.result()
                raise RuntimeError("unexpected error")

        # 和普通 step 一样，先吃掉执行期间积累的 abort，再推进 scheduler。
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): 延迟任务既可以在这里直接处理，也可以暂存在字段里，
        # 等 step_with_batch_queue 下次再被调用时立刻处理。后者会略微偏向
        # TTFT，而不是 TPOT/整体吞吐。
        if deferred_scheduler_output:
            # 如果启用了带 structured output 的 speculative decoding，
            # 就必须先取到上一轮的 draft token ids，才能为延迟请求计算
            # grammar bitmask。
            if self.use_spec_decode:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                assert draft_token_ids is not None
                # 把 scheduler 输出里的 draft token ids 更新掉，
                # 这样无效的 speculative token 会被填成 -1，
                # 后续 grammar bitmask 计算时会自动跳过。
                self.scheduler.update_draft_token_ids_in_output(
                    draft_token_ids, deferred_scheduler_output
                )
            # 现在已经拿到了计算延迟请求 bitmask 所需的 token，
            # 接着取 bitmask 并调用 sample_tokens。
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        return engine_core_outputs, model_executed

    def _process_aborts_queue(self):
        # socket 线程可能在模型执行时不断收到 abort。
        # 这里把这一小段时间内的 abort 全部合并成一个批次，一次性处理。
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                # 正常应该是 list，这里顺手兼容一下 string。
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
            # 合并成一个批次统一 abort 会更高效。
            self.abort_requests(request_ids)

    def shutdown(self):
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        self.model_executor.profile(is_start, profile_prefix)

    def reset_mm_cache(self):
        # NOTE: 这个接口主要用于调试，因此这里不尝试重新同步
        # 内部缓存状态（P0 sender、P1 receiver）。
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # cache 可能存在于 EngineCore，也可能存在于 WorkerWrapperBase。
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        """重置 encoder cache，使所有缓存的 encoder 输出失效。

        当模型权重发生更新时，应调用此方法，避免继续复用基于旧权重计算出的
        视觉 embedding。它会同时清理 scheduler 的缓存管理器和 GPU model
        runner 的缓存。
        """
        # NOTE: 这个接口主要用于调试，因此这里不尝试重新同步
        # 内部缓存状态（P0 sender、P1 receiver）。
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the encoder cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # 重置 scheduler 的 encoder cache 管理器（逻辑状态）
        self.scheduler.reset_encoder_cache()
        # 重置 GPU model runner 的 encoder cache（物理存储）
        self.model_executor.reset_encoder_cache()

    def _reset_caches(self, reset_running_requests=True) -> None:
        self.reset_prefix_cache(reset_running_requests=reset_running_requests)
        self.reset_mm_cache()
        self.reset_encoder_cache()

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """暂停生成；具体行为取决于模式。

        所有暂停模式都会把新的 add 请求先排队；其中 `abort` 和 `keep`
        会跳过 `step()`，而 `wait` 允许继续 `step()`，让飞行中的请求自然排空。

        - `abort`：设置为 `PAUSED_NEW`，终止全部请求，等待 abort 输出发完
          （在启用 `output_queue` 时），可选清空缓存，然后完成返回的 Future。
        - `wait`：设置为 `PAUSED_NEW`（新请求排队，但继续执行 step）；等请求
          自然排空后，可选清空缓存，然后完成返回的 Future。
        - `keep`：设置为 `PAUSED_ALL`；返回一个 Future，在输出队列清空后完成。
        """
        if mode not in ("keep", "abort", "wait"):
            raise ValueError(f"Invalid pause mode: {mode}")
        if mode == "wait":
            raise ValueError("'wait' mode can't be used in inproc-engine mode")

        if mode == "abort":
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)
        if clear_cache:
            self._reset_caches()

        return None

    def resume_scheduler(self) -> None:
        """恢复 scheduler，并继续处理暂停期间积压的请求。"""
        self.scheduler.set_pause_state(PauseState.UNPAUSED)

    def is_scheduler_paused(self) -> bool:
        """返回 scheduler 当前是否处于任意暂停状态。"""
        return self.scheduler.pause_state != PauseState.UNPAUSED

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None | Future:
        """按指定级别让引擎进入休眠。

        Args:
            level: 休眠级别。
                - Level 0：只暂停调度。仍然接收请求，但不处理；GPU 显存不变。
                - Level 1：把模型权重卸载到 CPU，并丢弃 KV cache。
                - Level 2：释放全部 GPU 显存。
            mode: 暂停模式，表示如何处理现有请求；详见 `pause_scheduler`
                的文档说明。
        """

        # 进入休眠前先暂停 scheduler。
        clear_prefix_cache = level >= 1
        pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
        if level < 1:
            return pause_future

        # Level 1 及以上：把 GPU 显存管理交给 executor 处理
        model_executor = self.model_executor
        if pause_future is None:
            model_executor.sleep(level)
            return None

        future = Future[Any]()

        def pause_complete(f: Future):
            try:
                f.result()  # 透传任何异常
                future.set_result(model_executor.sleep(level))
            except Exception as e:
                future.set_exception(e)

        logger.info("Waiting for in-flight requests to complete before sleeping...")
        pause_future.add_done_callback(pause_complete)
        return future

    def wake_up(self, tags: list[str] | None = None):
        """把引擎从休眠状态唤醒。

        Args:
            tags: 要唤醒的标签。Level 0 唤醒可传 `["scheduling"]`。
        """
        if tags is not None and "scheduling" in tags:
            # 如果还有其他标签要处理，就把 "scheduling" 从 tags 中移除。
            tags = [t for t in tags if t != "scheduling"]

        if tags is None or tags:
            self.model_executor.wake_up(tags)

        # 恢复调度（对所有 level 都适用）
        self.resume_scheduler()

    def is_sleeping(self) -> bool:
        """检查引擎是否处于任意级别的休眠状态。"""
        return self.is_scheduler_paused() or self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """对请求做预处理。

        这个函数可以直接在输入处理线程中调用，从而让请求初始化与模型前向并行进行。
        """
        # 线程安全说明：这里不存在竞态条件。
        # `mm_receiver_cache` 会在 LLMEngine 初始化末尾被重置，
        # 之后只会在输入处理线程中访问。
        # 多模态场景下，先把接收到的 mm feature 和本地缓存状态对齐。
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        # 把前端传来的 EngineCoreRequest 转成 scheduler 真正使用的 Request。
        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # 线程安全说明：这里不存在竞态条件。
            # `grammar_init` 只会在输入处理线程中被调用。对于
            # `structured_output_manager`，每个请求彼此独立，且 grammar
            # 编译是异步的。Scheduler 在调度请求前总会检查 grammar 的编译状态。
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave

    def _eep_scale_up_before_kv_init(self):
        raise NotImplementedError

    def _eep_send_engine_core_notification(
        self,
        notification_type: EEPNotificationType,
        vllm_config: VllmConfig | None = None,
    ):
        raise NotImplementedError


class EngineCoreProc(EngineCore):
    # `EngineCoreProc` = `EngineCore` + 后台进程通信外壳。
    # 它本身仍然继承 `EngineCore` 负责调度和执行；
    # 额外增加：
    # 1. 输入线程：把 ZMQ 消息解码后送入 input_queue
    # 2. 输出线程：把 EngineCoreOutputs 从 output_queue 编码后发回前端
    # 3. busy loop：消费 input_queue，并不断调用 step_fn()
    """用于在后台进程中运行 EngineCore 的 ZMQ 封装层。"""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"
    addresses: EngineZmqAddresses

    @instrument(span_name="EngineCoreProc init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        *,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False

        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            self.client_count = len(addresses.outputs)

            # 先通过握手拿到和前端/协调器通信所需的 ZMQ 地址，
            # 再决定自己是不是运行在 DP 协调场景中。
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            internal_dp_balancing = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )
            # 只有在 "internal" 和 "hybrid" 负载均衡模式下，
            # 才向 coordinator 上报请求队列统计信息。
            self.publish_dp_lb_stats = internal_dp_balancing

            self.addresses = addresses
            self.process_input_queue_block = True
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                self._eep_send_engine_core_notification(
                    EEPNotificationType.NEW_CORE_ENGINES_INIT_READY,
                    vllm_config=vllm_config,
                )
            self._init_data_parallel(vllm_config)

            super().__init__(
                vllm_config,
                executor_class,
                log_stats,
                executor_fail_callback,
                internal_dp_balancing,
            )

            # IO 和核心调度主循环分线程：
            # socket 线程负责 ZMQ <-> queue，主线程只处理 queue，
            # 这样可以把网络 IO / 序列化 和 GPU 前向尽量并行起来。
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # 在收到 DP coordinator 的 ready 消息之前，不要结束握手流程。
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """执行启动握手。

        对于 DP=1 或离线模式，握手对象是本机同置的 front-end 进程。

        对于启用内部负载均衡的 DP>1 场景，握手对象是共享的 front-end
        进程，它可能运行在另一台机器上。

        对于启用外部或混合负载均衡的 DP>1 场景，需要执行两次握手：
        - 与 rank 0 的 front-end 进程握手，获取 DP Coordinator 的 ZMQ 地址
          和 DP process group 地址；
        - 与本机同置的 front-end 进程握手，获取 client 的 input/output
          socket 地址。

        其中 rank 0 的 engine 和本机同置的 engine 本身不需要第二次握手。

        这里的 “front-end” 进程，既可能是持有 engine core client 的进程
        （例如 API server 未做横向扩展时的 API server 进程），也可能是
        `serve.py` 中运行 `run_multi_api_server()` 的 launcher 进程。
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            with handshake as addresses:
                yield addresses
        else:
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # 把握手阶段可能发生变化的配置重新同步到 vllm_config。
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # 向 front-end 注册 engine。
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # 发送 ready 消息。
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            # 对于 external LB 场景，这里会把 coordinator 的统计上报地址
            # 回传给本机同置的 front-end 使用（coordinator 只跑在 rank 0）。
            dp_stats_address = self.frontend_stats_publish_address

            # 带上配置 hash，用于校验 DP 配置是否一致
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
                "num_gpu_blocks": num_gpu_blocks,
                "dp_stats_address": dp_stats_address,
            }
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # 启动时先给前端发 HELLO，前端返回 input/output/coordinator 等地址，
        # 后台进程之后就按这些地址建立真正的数据通道。
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # 接收初始化消息。
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """在后台进程中启动 EngineCore 的 busy loop。"""

        # 用于优雅退出的信号处理器。
        # SystemExit 只会抛出一次，确保当前进程和 worker 进程都能无错误退出。
        shutdown_requested = False

        # 确保 spawn 之后仍然可以正确序列化 transformer config
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # SIGTERM 或 SIGINT 都会终止 engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: EngineCoreProc | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
                maybe_init_worker_tracer(
                    instrumenting_module_name="vllm.engine_core",
                    process_kind="engine_core",
                    process_name=f"EngineCore_DP{dp_rank}",
                )
                set_process_title("EngineCore", f"DP{dp_rank}")
            else:
                maybe_init_worker_tracer(
                    instrumenting_module_name="vllm.engine_core",
                    process_kind="engine_core",
                    process_name="EngineCore",
                )
                set_process_title("EngineCore")
            decorate_logs()

            if data_parallel and vllm_config.kv_transfer_config is not None:
                # 修改 engine_id，并把 local_dp_rank 追加进去，
                # 以保证每个 DP rank 的 kv_transfer_config 都是唯一的。
                vllm_config.kv_transfer_config.engine_id = (
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
                )
                logger.debug(
                    "Setting kv_transfer_config.engine_id to %s",
                    vllm_config.kv_transfer_config.engine_id,
                )

            parallel_config.data_parallel_index = dp_rank
            if data_parallel and vllm_config.model_config.is_moe:
                # 为当前 engine 进程设置数据并行 rank。
                parallel_config.data_parallel_rank = dp_rank
                # MoE + DP 需要跨 rank 协同，所以走 DPEngineCoreProc。
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                # 非 MoE 的 DP rank 之间彼此独立，不需要 wave 协同，
                # 所以直接按普通 EngineCoreProc 处理。
                # 注意：parallel_config.data_parallel_index 仍会保留原始 DP rank。
                parallel_config.data_parallel_size = 1
                parallel_config.data_parallel_size_local = 1
                parallel_config.data_parallel_rank = 0
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            assert engine_core is not None
            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def has_work(self) -> bool:
        """返回当前引擎是否需要继续执行 step。"""
        return (
            self.engines_running
            or self.scheduler.has_requests()
            or bool(self.batch_queue)
        )

    def run_busy_loop(self):
        """EngineCore 的核心 busy loop。"""

        # 持续循环，直到进程收到 SIGINT 或 SIGTERM
        while True:
            # 1. 先把前端发来的请求吃进来，必要时阻塞等待工作。
            self._process_input_queue()
            # 2. 再推进一轮 EngineCore，并把结果写回 output_queue。
            self._process_engine_step()

    def _process_input_queue(self):
        """当需要执行一次 engine step 时退出。"""

        waited = False
        while not self.has_work():
            # 完全空闲时，先通知那些在等“引擎已空闲”的回调。
            self._notify_idle_state_callbacks()
            if self.input_queue.empty():
                # 清空 aborts queue；所有 abort 同时也会经由 input_queue 处理。
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True
            block = self.process_input_queue_block
            try:
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
            except queue.Empty:
                break
            if not block:
                break

        if waited:
            logger.debug("EngineCore loop active.")

        # 一旦准备进入 step，再顺手把当前队列里已经到达的请求全部清空，
        # 这样本轮调度看到的状态更完整。
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """只会在当前本地仍有未完成请求时被调用。"""

        # 执行一次核心调度循环。
        outputs, model_executed = self.step_fn()
        # 一个 step 里可能产生多个 client 的输出，逐个写入输出队列。
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # step 执行后的钩子。
        self.post_step(model_executed)

        # 如果本轮没有执行模型，但仍然有等待中的请求
        # （例如 WAITING_FOR_REMOTE_KVS），那就短暂让出 GIL，
        # 给后台线程（例如 NIXL 握手线程）一点推进空间。
        # 否则紧凑的轮询循环可能会饿死这些后台线程。
        if not model_executed and self.scheduler.has_unfinished_requests():
            time.sleep(0.001)

        return model_executed

    def _notify_idle_state_callbacks(self) -> None:
        while self._idle_state_callbacks:
            callback = self._idle_state_callbacks.pop()
            callback(self)

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """分发来自 client 的请求。"""

        if request_type == EngineCoreRequestType.ADD:
            # ADD 请求在 socket 线程里已经完成了大部分反序列化和预处理，
            # 这里直接交给 scheduler。
            req, request_wave = request
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            # utility 请求不是正常推理，而是调用 EngineCore 的辅助方法，
            # 例如 reset_cache / add_lora / sleep / get_supported_tasks。
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            # 延迟查找 utility 方法，这样查找失败也能被统一处理并返回。
            get_result = lambda: (method := getattr(self, method_name)) and method(
                *self._convert_msgspec_args(method, args)
            )
            enqueue_output = lambda out: self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=out))
            )
            self._invoke_utility_method(method_name, get_result, output, enqueue_output)
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    @staticmethod
    def _invoke_utility_method(
        name: str, get_result: Callable, output: UtilityOutput, enqueue_output: Callable
    ):
        try:
            result = get_result()
            if isinstance(result, Future):
                # 等 future 完成后再处理 utility 输出。
                callback = lambda future: EngineCoreProc._invoke_utility_method(
                    name, future.result, output, enqueue_output
                )
                result.add_done_callback(callback)
                return
            output.result = UtilityResult(result)
        except Exception as e:
            logger.exception("Invocation of %s method failed", name)
            output.failure_message = f"Call to {name} method failed: {str(e)}"
        enqueue_output(output)

    @staticmethod
    def _convert_msgspec_args(method, args):
        """如果传入参数类型与目标方法声明类型不匹配，尝试把它转换成 msgspec 对象。"""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """向 EngineCoreClient 发送 EngineDead 状态。"""

        # 把 ENGINE_CORE_DEAD 放进输出队列。
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # 在 shutdown 前等待 daemon 把消息真正发出去。
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """输入 socket 的 IO 线程。"""

        # 这个线程专门负责：
        # 1. 从前端 socket 收消息
        # 2. 反序列化请求
        # 3. 对 ADD 请求做预处理
        # 4. 放入 input_queue 供主 busy loop 消费
        #
        # 这么做可以把部分 CPU 侧工作和 GPU 前向并行起来。
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # 向 coordinator 发送订阅消息。
                coord_socket.send(b"\x01")

            # 把 socket 注册到 poller。
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # 先给每个 input socket 发一个初始消息；
                # 这是 front-end 的 ROUTER socket 后续能把消息发回来的前提。
                input_socket.send(b"")
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # 等待 coordinator 的 ready 消息。
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # （请求类型，请求数据）
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    # NOTE(yongji): 忽略 DP coordinator 发来的 READY 消息，
                    # 这个消息只用来通知新启动的 engine。
                    if type_frame.buffer == b"READY":
                        assert input_socket == coord_socket
                        continue
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # 按 request type 做对应的反序列化和前处理。
                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    else:
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            # abort 同时进两个队列：
                            # 1. aborts_queue：让一次 forward 结束后尽快处理
                            # 2. input_queue：保证和普通请求之间的时序关系
                            # scheduler 的 abort 是幂等的，所以这样做没问题。
                            self.aborts_queue.put_nowait(request)

                    # 统一交给主 busy loop 做最终分发。
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self,
        output_paths: list[str],
        coord_output_path: str | None,
        engine_index: int,
    ):
        """输出 socket 的 IO 线程。"""

        # 和输入线程相反，这里负责：
        # 1. 从 output_queue 取 EngineCoreOutputs
        # 2. 做序列化
        # 3. 通过对应 socket 发回前端/协调器
        encoder = MsgpackEncoder()
        # 可复用的发送缓冲区。
        reuse_buffers: list[bytearray] = []
        # 在 zmq 真正发送完成前，保留 outputs 和 buffer 的引用
        # （outputs 里可能含有 tensor/np array，其底层 buffer 会被提取出来做零拷贝发送）。
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # 必须设置 linger，确保在关闭 socket 前能把 ENGINE_CORE_DEAD 发出去。
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # `-1` 不是普通前端 client，而是发给 coordinator 的控制/统计消息。
                    # coordinator 消息通常很小，不必复用 buffer。
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # 回收那些 zmq 已经发送完成的 buffer。
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # 限制可复用 buffer 的数量。
                    reuse_buffers.append(buffer)

    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
        """记录并返回按请求粒度组织的错误响应。

        这里处理的是输入 socket 线程里，ADD 请求预处理阶段抛出的异常。
        """
        logger.exception(
            "Unexpected error pre-processing request %s", request.request_id
        )
        self.output_queue.put_nowait(
            (
                request.client_index,
                EngineCoreOutputs(
                    engine_index=self.engine_index,
                    finished_requests={request.request_id},
                    outputs=[
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[],
                            finish_reason=FinishReason.ERROR,
                        )
                    ],
                ),
            )
        )

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """暂停生成；具体行为取决于模式。

        所有暂停模式都会把新的 add 请求先排队；其中 `abort` 和 `keep`
        会跳过 `step()`，而 `wait` 允许继续 `step()`，让飞行中的请求自然排空。

        - `abort`：设置为 `PAUSED_NEW`，终止全部请求，等待 abort 输出发完
          （在启用 `output_queue` 时），可选清空缓存，然后完成返回的 Future。
        - `wait`：设置为 `PAUSED_NEW`（新请求排队，但继续执行 step）；等请求
          自然排空后，可选清空缓存，然后完成返回的 Future。
        - `keep`：设置为 `PAUSED_ALL`；返回一个 Future，在输出队列清空后完成。
        """
        if mode not in ("keep", "abort", "wait"):
            raise ValueError(f"Invalid pause mode: {mode}")

        def engine_idle_callback(engine: "EngineCoreProc", future: Future[Any]) -> None:
            if clear_cache:
                engine._reset_caches()
            future.set_result(None)

        if mode == "abort":
            aborted_reqs = self.scheduler.finish_requests(
                None, RequestStatus.FINISHED_ABORTED
            )
            self._send_abort_outputs(aborted_reqs)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)
        if not self.has_work():
            if clear_cache:
                self._reset_caches()
            return None

        future = Future[Any]()
        self._idle_state_callbacks.append(partial(engine_idle_callback, future=future))
        return future

    def _send_abort_outputs(self, aborted_reqs: list[tuple[str, int]]) -> None:
        # TODO(nick): 这段逻辑后续会移到 scheduler 内部
        # 这里把“哪些请求被 abort 了”重新按 client 分组，组装成
        # EngineCoreOutputs 发回去，避免前端一直等不到 finished 信号。
        if aborted_reqs:
            # 建立 client_index 到其所属 request_id 列表的映射。
            by_client = defaultdict[int, set[str]](set)
            for req_id, client_index in aborted_reqs:
                by_client[client_index].add(req_id)
            for client_index, req_ids in by_client.items():
                outputs = [
                    EngineCoreOutput(req_id, [], finish_reason=FinishReason.ABORT)
                    for req_id in req_ids
                ]
                eco = EngineCoreOutputs(finished_requests=req_ids, outputs=outputs)
                self.output_queue.put_nowait((client_index, eco))


class DPEngineCoreProc(EngineCoreProc):
    # MoE + DP 专用版本。
    # 在普通 EngineCoreProc 的基础上，额外维护：
    # 1. `current_wave`：当前这轮 DP 协同批次
    # 2. `engines_running`：全局各 rank 是否仍有未完成请求
    # 3. request count / step counter：给 coordinator 做内部负载均衡与状态同步
    """用于在数据并行场景下、后台进程中运行 EngineCore 的 ZMQ 封装层。"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
    ):
        assert vllm_config.model_config.is_moe, (
            "DPEngineCoreProc should only be used for MoE models"
        )

        # 统计模型前向执行次数，用于每隔 N 步与 DP peer 同步一次完成状态。
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        self.eep_scaling_state: ElasticEPScalingState | None = None

        # 初始化引擎。
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            engine_index=dp_rank,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # 为数据并行配置 GPU 和无状态 process group。
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self.dp_rank = dp_rank
        self.dp_group, self.dp_store = (
            vllm_config.parallel_config.stateless_init_dp_group(return_store=True)
        )

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def add_request(self, request: Request, request_wave: int = 0):
        super().add_request(request, request_wave)
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running:
                # 当前 wave 已经结束，但前端又给了旧 wave 的请求，
                # 说明需要显式通知前端/协调器启动下一轮。
                # 收到了属于已完成 wave 的请求，需要通知 front-end
                # 启动下一轮 wave。
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

    def resume_scheduler(self):
        super().resume_scheduler()
        if (
            self.has_coordinator
            and not self.engines_running
            and self.scheduler.has_unfinished_requests()
        ):
            # 唤醒其他 DP engine。
            self.output_queue.put_nowait(
                (-1, EngineCoreOutputs(start_wave=self.current_wave))
            )

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            # 协调器要求所有 DP rank 从某个新 wave 开始恢复执行。
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave %d.", new_wave)
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # 把本 rank 当前 waiting/running 数上报给 coordinator，
        # 供前端做内部负载均衡决策。
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, step_counter=self.step_counter, current_wave=self.current_wave
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def run_busy_loop(self):
        """数据并行场景下的 EngineCore 核心 busy loop。"""

        # 持续循环，直到进程收到 SIGINT 或 SIGTERM
        while True:
            # 1) 轮询 input queue，直到出现可处理的工作。
            self._process_input_queue()

            if self.eep_scaling_state is not None:
                _ = self.eep_scaling_state.progress()
                if self.eep_scaling_state.is_complete():
                    self.process_input_queue_block = True
                    self.eep_scaling_state = None

            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # 所有 engine 都处于空闲状态。
                    continue

                # 只要全局还处于 running wave，就算本 rank 这一轮没有可执行请求，
                # 也要跑 dummy batch，避免不同 rank 的执行节拍脱离。
                self.execute_dummy_batch()

            # 每隔若干步做一次跨 DP rank 同步，判断“全局是否仍有未完成请求”。
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # 通知 client：当前循环即将暂停。
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # 如果有 coordinator，则由 dp rank 0 向 coordinator 发更新；
                    # 否则（离线 SPMD 场景）每个 rank 都向自己同置的 front-end
                    # 进程发更新。
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # 当前 wave 完整结束，切到下一 wave。
                self.current_wave += 1
                self.step_counter = 0

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # 这是一个性能优化：不是每步都 all-reduce，而是每 32 步同步一次。
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        from copy import deepcopy

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        new_parallel_config = deepcopy(self.vllm_config.parallel_config)
        old_dp_size = new_parallel_config.data_parallel_size
        new_parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            new_parallel_config.data_parallel_rank = (
                reconfig_request.new_data_parallel_rank
            )
        new_parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        new_parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        new_parallel_config._data_parallel_master_port_list = (
            reconfig_request.new_data_parallel_master_port_list
        )

        is_scale_down = reconfig_request.new_data_parallel_size < old_dp_size
        is_shutdown = (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        )

        self.eep_scaling_state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=new_parallel_config,
            worker_type="removing" if is_shutdown else "existing",
            scale_type="scale_down" if is_scale_down else "scale_up",
            reconfig_request=reconfig_request,
        )
        self.process_input_queue_block = False
        logger.info(
            "[Elastic EP] Received reconfiguration request and starting scaling up/down"
        )

    def _eep_send_engine_core_notification(
        self,
        notification_type: EEPNotificationType,
        vllm_config: VllmConfig | None = None,
    ):
        """向 EngineCoreClient 发送通知，由它继续转发给其他 engine core 进程。

        这个机制用于：
        1. scale up 时：新加入的 core engine 通知已有 core engine 自己已就绪；
        2. scale down 时：即将移除的 core engine 通知 EngineCoreClient，
           让它释放对应的 Ray placement group；
        3. scale up/down 两种场景：通知 EngineCoreClient，已有 core engine
           已切换到新的并行配置。
        """
        if vllm_config is None:
            dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        else:
            dp_rank = vllm_config.parallel_config.data_parallel_rank
        notification_data = (notification_type.value, dp_rank)
        outputs = EngineCoreOutputs(
            utility_output=UtilityOutput(
                call_id=EEP_NOTIFICATION_CALL_ID,
                result=UtilityResult(notification_data),
            )
        )
        outputs.engine_index = self.engine_index

        if hasattr(self, "output_thread") and self.output_thread.is_alive():
            self.output_queue.put_nowait((0, outputs))
        else:
            encoder = MsgpackEncoder()
            with (
                zmq.Context() as ctx,
                make_zmq_socket(
                    ctx, self.addresses.outputs[0], zmq.PUSH, linger=4000
                ) as socket,
            ):
                socket.send_multipart(encoder.encode(outputs))

    def eep_handle_engine_core_notification(
        self, notification_type: str | EEPNotificationType
    ):
        """处理从 EngineCoreClient 收到的通知。

        这些通知本质上是由新加入的 core engine 转发过来的。
        """
        assert self.eep_scaling_state is not None
        if isinstance(notification_type, str):
            notification_type = EEPNotificationType(notification_type)
        self.eep_scaling_state.handle_notification(notification_type)

    def _eep_scale_up_before_kv_init(self):
        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        self.eep_scaling_state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=self.vllm_config.parallel_config,
            worker_type="new",
            scale_type="scale_up",
            reconfig_request=None,
        )
        self.model_executor.collective_rpc("init_device")
        self.model_executor.collective_rpc("load_model")
        self._eep_send_engine_core_notification(
            EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY
        )
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("receive_weights",)
        )
        self.available_gpu_memory_for_kv_cache = (
            ParallelConfig.sync_kv_cache_memory_size(self.dp_group, -1)
        )
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("prepare_new_worker",)
        )
        self.process_input_queue_block = False


class EngineCoreActorMixin:
    """用于在数据并行场景下运行 EngineCore 的 Ray actor 混入类。"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        # 如果配置了分布式 tracing，就在这里初始化 tracer。
        maybe_init_worker_tracer(
            instrumenting_module_name="vllm.engine_core",
            process_kind="engine_core",
            process_name=f"DPEngineCoreActor_DP{dp_rank}",
        )

        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_index = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        # 尽可能早地在 actor 生命周期里设置 CUDA_VISIBLE_DEVICES。
        # NOTE: 在多进程模式下，这个变量会在进程创建时设置；
        # 但在 Ray 里不能同样处理，因为：
        # 1) Ray 管理所有 ray worker 的生命周期（包括 DPEngineCoreActor）；
        # 2) Ray 会根据 num_gpus 配置自动设置 CUDA_VISIBLE_DEVICES。
        # 为了绕过第 2 点，还需要设置
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES；但这样之后创建出的
        # vLLM worker 会继承一个“粘住”的 CUDA_VISIBLE_DEVICES：
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # 这会带来问题：当 vLLM worker（一个 Ray actor）执行任务时，
        # 它会按这个“粘住”的 CUDA_VISIBLE_DEVICES 去索引，而不是直接用 GPU ID，
        # 从而可能触发索引越界。参见：
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # 以及 Ray 的 worker.py 里
        # `get_accelerator_ids_for_accelerator_resource()` 的实现。
        self._set_visible_devices(vllm_config, local_dp_rank)

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            self._set_cuda_visible_devices(
                vllm_config, local_dp_rank, device_control_env_var
            )

    def _set_cuda_visible_devices(
        self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
    ):
        world_size = vllm_config.parallel_config.world_size
        # 设置 CUDA_VISIBLE_DEVICES 或等价的设备控制环境变量。
        try:
            value = get_device_indices(
                device_control_env_var, local_dp_rank, world_size
            )
            os.environ[device_control_env_var] = value
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        Ray 场景下，地址信息在 actor 创建前就已经准备好了，
        不需要像多进程本地模式那样再走一遍 ZMQ startup handshake。
        """
        yield self.addresses

    def wait_for_init(self):
        """等待 engine core 完成初始化。

        这个方法本身什么都不做。只要对它（或 actor 的任意其他方法）执行
        `ray.get()` 并成功返回，就说明 actor 创建流程（也就是 `__init__`）
        已经完成。
        """
        pass

    def run(self):
        """运行 engine core 的 busy loop。"""
        try:
            self.run_busy_loop()  # type: ignore[attr-defined]
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()  # type: ignore[attr-defined]


class DPMoEEngineCoreActor(EngineCoreActorMixin, DPEngineCoreProc):
    """用于 MoE 模型的数据并行场景。"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_rank = dp_rank

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        DPEngineCoreProc.__init__(
            self, vllm_config, local_client, "", executor_class, log_stats
        )


class EngineCoreActor(EngineCoreActorMixin, EngineCoreProc):
    """用于非 MoE 和/或非数据并行场景。"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.data_parallel_size_local = 1
        vllm_config.parallel_config.data_parallel_rank = 0

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        EngineCoreProc.__init__(
            self,
            vllm_config,
            local_client,
            "",
            executor_class,
            log_stats,
            engine_index=dp_rank,
        )
