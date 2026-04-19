# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object


@dataclass
class NewRequestData:
    """scheduler 第一次调度某个请求时，发送给 worker 的完整请求数据。"""

    # 请求 ID，worker 后续会用它作为缓存状态的 key。
    req_id: str
    # prompt token ids；如果请求使用 prompt_embeds，则这里可能为 None。
    prompt_token_ids: list[int] | None
    # 请求携带的多模态特征，例如图片、音频等输入描述。
    mm_features: list[MultiModalFeatureSpec]
    # 生成请求的采样参数；pooling 请求通常为 None。
    sampling_params: SamplingParams | None
    # pooling/embedding 请求的参数；生成请求通常为 None。
    pooling_params: PoolingParams | None
    # 该请求已经分配到的 KV cache block ids，按 KV cache group 分组。
    block_ids: tuple[list[int], ...]
    # 该请求当前已经完成 forward 计算的 token 数。
    num_computed_tokens: int
    # 该请求绑定的 LoRA adapter；没有则为 None。
    lora_request: LoRARequest | None
    # 调用方直接提供的 prompt embeddings；有它时 prompt_token_ids 可能为空。
    prompt_embeds: "torch.Tensor | None" = None

    # 仅 v2 model runner 使用：本轮 prefill 实际要处理的 token ids。
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "NewRequestData":
        # 把 scheduler 内部的 Request 对象转换成可发送给 worker 的数据对象。
        # worker 侧会缓存这些完整信息，后续调度同一请求时只需要发送增量。
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prefill_token_ids=prefill_token_ids,
        )

    def __repr__(self) -> str:
        # repr 中只展示 prompt_embeds 的 shape，避免直接打印大 tensor。
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # 脱敏版 __repr__：隐藏 prompt token 内容，只展示长度，避免日志泄露用户输入。
    def anon_repr(self) -> str:
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


@dataclass
class CachedRequestData:
    """scheduler 再次调度已缓存请求时，发送给 worker 的增量数据。"""

    # 本轮被调度的已缓存请求 ID 列表。
    req_ids: list[str]
    # 对不在 resumed_req_ids 中的请求，new_block_ids 会追加到已有 block ids 后面。
    # 对在 resumed_req_ids 中的请求，new_block_ids 会替换已有 block ids；
    # 这通常对应被抢占后恢复执行的请求。
    resumed_req_ids: set[str]
    # new_token_ids 仅 pipeline parallel 使用。
    # 不使用 PP 时，这个列表为空，因为普通路径 worker 已经能从本地状态拿到 token。
    new_token_ids: list[list[int]]
    # 对上一轮未被调度的请求，把完整 token ids 传给 connector 使用。
    # 上一轮已经被调度过的请求不会出现在这里，因为 worker/connector 已有连续状态。
    all_token_ids: dict[str, list[int]]
    # 每个请求新增或恢复后的 block ids；None 表示该请求本轮没有新增 block。
    new_block_ids: list[tuple[list[int], ...] | None]
    # 每个请求当前已经完成 forward 计算的 token 数。
    num_computed_tokens: list[int]
    # 每个请求当前已经生成的输出 token 数。
    num_output_tokens: list[int]

    # 脱敏版 dataclass repr：隐藏 token 内容，只展示 token 数量。
    def anon_repr(self) -> str:
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()
        }
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    def __repr__(self) -> str:
        return self.anon_repr()

    @property
    def num_reqs(self) -> int:
        # 已缓存请求数量就是 req_ids 的长度。
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """缓存 req_id 到 num_output_tokens 的映射，便于 O(1) 查询。

        这里使用 cached_property 是安全的，因为 CachedRequestData 每轮调度都会新建，
        并且在本轮调度细节计算过程中不会被修改。
        """
        return dict(zip(self.req_ids, self.num_output_tokens))

    def is_context_phase(self, req_id: str) -> bool:
        # output token 数为 0 表示该请求还处于 context/prefill 阶段。
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        # 构造一个没有任何已缓存请求的空增量对象，供空调度结果复用。
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@dataclass
class SchedulerOutput:
    """scheduler 每一轮调度后发送给 worker/model runner 的完整执行计划。"""

    # 首次被调度的请求列表。
    # worker 会缓存这些请求的完整数据，因此后续 step 不需要重复发送完整请求。
    scheduled_new_reqs: list[NewRequestData]
    # 之前已经调度过、worker 侧已有缓存的请求。
    # 因为完整请求数据已经在 worker 里，所以这里只发送增量，降低通信成本。
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens。
    # 表示每个请求本轮被安排执行多少个 token。
    num_scheduled_tokens: dict[str, int]
    # 本轮所有请求合计调度的 token 数。
    # 等于 sum(num_scheduled_tokens.values())。
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids。
    # 如果某个请求本轮没有 speculative decode tokens，就不会出现在该字典中。
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> 本轮需要处理的 encoder input 下标列表。
    # 例如某请求对应 [0, 1]，可能表示本轮 vision encoder 需要处理该请求的
    # 第 0 张和第 1 张图片。
    scheduled_encoder_inputs: dict[str, list[int]]
    # 每个 KV cache group 中，当前 batch 所有请求共享的公共前缀 block 数。
    # cascade attention 会用它来优化公共前缀处理。
    num_common_prefix_blocks: list[int]

    # 从上一轮到当前轮之间已经结束的请求 ID。
    # worker 用它释放这些请求对应的缓存状态。
    finished_req_ids: set[str]
    # 需要从 encoder cache 中释放的 encoder output 对应 mm_hash 列表。
    free_encoder_mm_hashes: list[str]

    # 本轮被抢占的请求 ID。仅 v2 model runner 使用。
    preempted_req_ids: set[str] | None = None

    # 本轮被调度的请求中，是否有请求使用 structured output。
    # 仅 async scheduling 场景设置。
    has_structured_output_requests: bool = False

    # 被调度请求是否还缺少 grammar bitmask 计算所需的输出 token。
    pending_structured_output_tokens: bool = False

    # 用于修正 speculative decoding acceptance rate 统计。
    num_invalid_spec_tokens: dict[str, int] | None = None

    # KV Cache Connector 使用的 metadata。
    kv_connector_metadata: KVConnectorMetadata | None = None

    # EC Cache Connector 使用的 metadata。
    ec_connector_metadata: ECConnectorMetadata | None = None

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        # 构造一个没有任何可执行 token 的空调度结果，供 worker 空转/无任务场景使用。
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


@dataclass
class GrammarOutput:
    """structured output grammar 约束的调度输出。"""

    # 使用 structured output 的请求 ID 列表。
    structured_output_request_ids: list[str]
    # grammar bitmask，顺序与 structured_output_request_ids 一一对应。
    grammar_bitmask: "npt.NDArray[np.int32]"
