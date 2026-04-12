# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.metrics.stats import SchedulerStats
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager


class PauseState(enum.IntEnum):
    """调度器的暂停状态。

    - UNPAUSED: 正常运行。
    - PAUSED_NEW: 不再调度新请求，但仍会继续调度已经处于
      running 状态的请求。
    - PAUSED_ALL: 不调度任何请求。
    """

    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


class SchedulerInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def schedule(self) -> "SchedulerOutput":
        """决定这一调度步要处理哪些请求。

        调度决策是在 iteration 粒度上做出的。每一个调度步基本对应模型
        的一次前向计算，因此这个方法会在引擎的 busy loop 中被反复调用。

        从本质上说，调度器会产出一个形如 {req_id: num_tokens} 的映射，
        表示这一轮中每个请求要处理多少个 token。比如，对新请求来说，
        num_tokens 可能大到等于整个 prompt 的长度；而对正在自回归、
        一次生成一个 token 的请求来说，num_tokens 可能就是 1。在
        chunked prefill、prefix cache、speculative decoding 等场景下，
        这个值也可能介于两者之间。

        此外，调度器还会返回一些关于单个请求或整个 batch 的有用信息，
        model runner 会用这些信息来准备模型输入。

        Returns:
            一个 SchedulerOutput 对象，其中包含本轮被调度请求的信息。
        """
        raise NotImplementedError

    @abstractmethod
    def get_grammar_bitmask(
        self, scheduler_output: "SchedulerOutput"
    ) -> "GrammarOutput | None":
        raise NotImplementedError

    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        """根据 model runner 的输出更新调度器状态。

        这个方法会在 model runner 处理完本轮调度请求之后调用。模型输出
        中可能包含新生成的 token id、下一轮要用的 draft token id 等
        信息。调度器会利用这些信息更新内部状态、检查哪些请求已经结束，
        并为每个请求生成对应的输出。

        Returns:
            一个从 client index 到 EngineCoreOutputs 的映射，包含各个
            client 发起请求的输出结果。
        """
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids(self, draft_token_ids: "DraftTokenIds") -> None:
        """用新生成的 draft token id 更新请求。

        如有需要，还会对 structured output 执行 grammar 校验。

        Args:
            draft_token_ids: 每个请求对应的 draft token id 输入。
        """
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids_in_output(
        self, draft_token_ids: "DraftTokenIds", scheduler_output: "SchedulerOutput"
    ) -> None:
        """用新生成的 draft token id 更新 scheduler output。

        如有需要，还会对 structured output 执行 grammar 校验。

        Args:
            draft_token_ids: 每个请求对应的 draft token id 输入。
            scheduler_output: 要被更新的 scheduler_output，会写入对应的
                draft token id。
        """
        raise NotImplementedError

    @abstractmethod
    def add_request(self, request: "Request") -> None:
        """向调度器内部队列加入一个新请求。

        Args:
            request: 要加入的新请求。
        """
        raise NotImplementedError

    @abstractmethod
    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: "RequestStatus",
    ) -> list[tuple[str, int]]:
        """结束调度器内部队列中的请求。

        如果某个请求不在队列里，这个方法不会对它做任何处理。

        这个方法通常会在两种情况下调用：
        1. 请求被客户端主动中止。
        2. 前端进程在将生成 token 反解码后，检测到了该请求的 stop string。

        Args:
            request_ids: 单个请求 ID、一组请求 ID，或 None。传 None 表示
                结束所有请求。
            finished_status: 这些请求要被设置成的结束状态。

        Returns:
            返回被本次中止的请求 (req_id, client_index) 元组列表。
            已经结束的请求不会重复包含在结果中。
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """返回调度器内部队列中尚未完成的请求数量。"""
        raise NotImplementedError

    def has_unfinished_requests(self) -> bool:
        """如果调度器内部队列中还有未完成请求，则返回 True。"""
        return self.get_num_unfinished_requests() > 0

    @abstractmethod
    def has_finished_requests(self) -> bool:
        """如果存在仍需清理的已结束请求，则返回 True。

        注意：这和 `not self.has_unfinished_requests()` 不是一回事。

        调度器会维护一个内部列表，记录上一步中刚刚结束的请求。这个列表
        会在下一次调用 schedule() 时返回出去，以便在下一步通知 model
        runner 清理这些已结束请求对应的缓存状态。

        这个方法就是用来检查这个“已结束请求内部列表”是否为空。
        这对 DP attention 场景很有用。
        """
        raise NotImplementedError

    def has_requests(self) -> bool:
        """如果还有未完成请求，或还有尚未通过 SchedulerOutput 返回出去的
        已结束请求，则返回 True。"""
        return self.has_unfinished_requests() or self.has_finished_requests()

    @property
    @abstractmethod
    def pause_state(self) -> PauseState:
        """当前调度器的暂停状态。"""
        raise NotImplementedError

    @abstractmethod
    def set_pause_state(self, pause_state: PauseState) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """重置 KV cache 的 prefix cache。

        当模型权重发生热更新时，这个操作尤其重要。

        Args:
            reset_running_requests: 如果为 True，所有正在运行的请求都会先被
                抢占并移回 waiting 队列。否则，只有在没有运行中请求占用
                KV cache 时，才会执行 KV prefix cache 的重置。
        """
        raise NotImplementedError

    @abstractmethod
    def reset_encoder_cache(self) -> None:
        """重置 encoder cache，使所有缓存过的 encoder 输出全部失效。

        当模型权重更新时，应调用这个方法，避免复用过期的视觉 embedding。
        """
        raise NotImplementedError

    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]:
        """返回 `(num_running_reqs, num_waiting_reqs)`。"""
        raise NotImplementedError

    @abstractmethod
    def make_stats(self) -> "SchedulerStats | None":
        """生成一个用于日志记录的 SchedulerStats 对象。

        这个 SchedulerStats 对象会在每一个调度步中创建。
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """关闭调度器。"""
        raise NotImplementedError

    def get_kv_connector(self) -> "KVConnectorBase_V1 | None":
        return None
