# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from typing_extensions import deprecated

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class StreamingUpdate:
    """Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None

    @classmethod
    def from_request(cls, request: "Request") -> "StreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
        )


class Request:
    """调度器内部使用的请求对象。

    这个类是 v1 调度路径里的核心状态载体。一个请求从进入 scheduler、
    被调度、执行、追加输出、命中/写入 prefix cache，到最终结束，相关状态
    都会持续写回到这个对象里。
    """

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: "LoRARequest | None" = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        resumable: bool = False,
        reasoning_ended: bool | None = None,
    ) -> None:
        # 请求身份与路由信息。
        # request_id: 请求的全局唯一标识。
        self.request_id = request_id
        # client_index: 该请求来自哪个前端 client，用于把结果路由回去。
        self.client_index = client_index
        # priority: 优先级调度时使用的优先级，值越小优先级越高。
        self.priority = priority

        # 请求的执行配置。
        # sampling_params: 文本生成请求的采样参数。
        self.sampling_params = sampling_params
        # pooling_params: pooling 请求的参数。与 sampling_params 二选一。
        self.pooling_params = pooling_params
        # lora_request: 该请求若使用 LoRA，这里保存对应的 LoRA 信息。
        self.lora_request = lora_request
        # structured_output_request: 若采样参数启用了结构化输出，这里保存对应
        # 的 grammar/FSM 请求对象；否则为 None。
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        if self.structured_output_request is not None:
            # reasoning_ended: 结构化输出场景下，用于标记 reasoning 阶段是否结束。
            self.structured_output_request.reasoning_ended = reasoning_ended
        # arrival_time: 请求进入系统的时间戳，用于 FCFS / priority 调度排序。
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        # 生命周期与事件信息。
        # status: 当前请求状态，决定它处于 waiting/running/preempted/finished
        # 的哪个阶段。
        self.status = RequestStatus.WAITING
        # events: 请求生命周期事件列表，用于日志、trace、观测。
        self.events: list[EngineCoreEvent] = []
        # stop_reason: 请求停止的具体原因，例如命中 stop token/string。
        self.stop_reason: int | str | None = None

        # P/D 场景下的 connector 特定参数，例如远端 KV 传输所需的上下文。
        self.kv_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # max_tokens: 请求最多生成多少个 token。
            # 对 pooling 模型没有“持续生成”过程，因此固定视作 1。
            self.max_tokens = 1
        elif sampling_params is not None:
            # 生成模型从 sampling_params 中读取 max_tokens。
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                # 结构化输出需要先等待 grammar/FSM 准备完成后才能调度。
                self.status = RequestStatus.WAITING_FOR_FSM

            if sampling_params.extra_args is not None:
                # extra_args 中可夹带 P/D 传输参数，供 connector 使用。
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        # Prompt / token 相关状态。
        # prompt_token_ids: 原始 prompt 的 token 序列；若走 prompt_embeds，则可能为 None。
        self.prompt_token_ids = prompt_token_ids
        # prompt_embeds: 直接输入模型的 embedding 形式 prompt。
        self.prompt_embeds = prompt_embeds
        # _prompt_embeds_per_block_hashes: prompt_embeds 按 block 切片后的 hash 缓存，
        # 避免重复为同一段 embedding 反复计算 hash。
        self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {}
        # num_prompt_tokens: prompt 的 token 长度。无论输入来自 token ids
        # 还是 prompt_embeds，都会统一折算成这个长度。
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        # _output_token_ids: 目前已经生成出来的输出 token。
        self._output_token_ids: list[int] = []
        # _all_token_ids: 请求当前完整 token 序列 = prompt + 已生成输出。
        # scheduler / prefix cache / block hash 计算都以它为基准。
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )

        # 异步调度相关状态。
        # num_output_placeholders: 异步调度时，为尚未真正落地的输出 token
        # 预留的占位符数量。
        self.num_output_placeholders = 0
        # discard_latest_async_tokens: 异步调度 + 强制抢占时，是否丢弃最近一次
        # 异步生成的 token，避免恢复后重复。
        self.discard_latest_async_tokens = False

        # speculative decoding / 调度进度 / prefix cache 相关状态。
        # spec_token_ids: 当前为该请求暂存的 speculative draft token。
        self.spec_token_ids: list[int] = []
        # num_computed_tokens: 已经完成模型计算的 token 数。它是 scheduler 里
        # 最关键的进度计数之一，用来表示“这个请求已经推进到哪里了”。
        self.num_computed_tokens = 0
        # cache_salt: 参与 prefix cache key/hash 计算的盐值，用来隔离缓存命名空间。
        self.cache_salt: str | None = cache_salt

        # 多模态相关状态。
        # mm_features: 多模态输入特征列表，例如图像编码后的位置信息等。
        self.mm_features = mm_features or []

        # 只读视图。
        # output_token_ids / all_token_ids 对外暴露为只读列表，避免调用方直接
        # append 导致内部多个 token 列表不一致。
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers: 链路追踪相关 header，会透传到输出或观测链路中。
        self.trace_headers = trace_headers

        # 调度与执行状态。
        # num_cached_tokens: 当前请求中命中 prefix cache 的 token 数。
        # -1 表示尚未统计/尚未初始化。
        self.num_cached_tokens = -1

        # is_prefill_chunk: 当前是否处在“prefill 被分块调度且尚未完成”的状态。
        # True 表示这只是 prefill 的中间分片，不是最后一个分片。
        self.is_prefill_chunk = False

        # num_nans_in_logits: 本请求对应 logits 中出现 NaN 的数量。
        # 大于 0 往往意味着输出已经损坏或模型执行异常。
        self.num_nans_in_logits = 0

        # num_preemptions: 该请求被 scheduler 抢占过多少次。
        # 这既会影响统计，也会影响某些缓存命中路径的处理方式。
        self.num_preemptions = 0

        # num_external_computed_tokens: 已由外部 connector / 远端 KV 侧计算好的
        # token 数，本地可直接复用其 KV。
        self.num_external_computed_tokens = 0

        # block_hashes: 按 block 粒度计算出的 hash 列表，用于 prefix cache 查询。
        self.block_hashes: list[BlockHash] = []
        # _block_hasher: 用于增量计算 block_hashes 的函数。
        # 这里不把 self 绑定进闭包，避免形成引用环，影响及时回收。
        self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
        self.update_block_hashes()

        # skip_reading_prefix_cache: 当前请求是否应跳过读取 prefix cache。
        # 常见于要求 prompt logprobs 或某些 pooling 路径。
        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        # 流式续写相关状态。
        # resumable: 该请求是否支持被视作一个“会话”并在后续继续追加输入。
        self.resumable = resumable
        # streaming_queue: 后续流式输入更新队列。队列里的每个元素都是一段新的
        # StreamingUpdate；其中 None 是结束哨兵，表示整个 streaming 会话结束。
        self.streaming_queue: deque[StreamingUpdate | None] | None = None

    @property
    @deprecated(
        "Request.eos_token_id will be removed in v0.18. "
        "Please use Request.sampling_params.eos_token_id instead."
    )
    def eos_token_id(self) -> int | None:
        if self.sampling_params is None:
            return None

        return self.sampling_params.eos_token_id

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            resumable=request.resumable,
            reasoning_ended=request.reasoning_ended,
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_features)

    @property
    def has_encoder_inputs(self) -> bool:
        return self.num_encoder_inputs > 0

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        return self.mm_features[input_id].mm_position.get_num_embeds()

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, arrival time, and request ID.
        Used in priority scheduling.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}
