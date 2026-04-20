# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import gc
import itertools
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    update_config,
)
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
    prepare_communication_buffer_for_model,
)
from vllm.forward_context import (
    BatchDescriptor,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    XDRotaryEmbedding,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsXDRoPE,
    is_mixture_of_experts,
    supports_eagle3,
    supports_mrope,
    supports_multimodal_pruning,
    supports_realtime,
    supports_transcription,
    supports_xdrope,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    is_pooling_model,
    is_text_generation_model,
)
from vllm.model_executor.offloader import (
    create_offloader,
    get_offloader,
    set_offloader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import (
    BatchedTensorInputs,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.tracing import instrument
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.nvtx_pytorch_hooks import PytHooks
from vllm.utils.platform_utils import is_pin_memory_available, num_compute_units
from vllm.utils.torch_utils import (
    get_dtype_size,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    create_fast_prefill_custom_backend,
    get_dcp_local_seq_lens,
    reorder_batch_to_split_decodes_and_prefills,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ECConnectorOutput,
    KVConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.cp_utils import (
    check_attention_cp_compatibility,
    get_total_cp_world_size,
)
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.ec_connector_model_runner_mixin import ECConnectorModelRunnerMixin
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    check_ubatch_thresholds,
    maybe_create_ubatch_slices,
    split_attn_metadata,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.workspace import lock_workspace

from .utils import (
    AttentionGroup,
    add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache,
    prepare_kernel_block_sizes,
    sanity_check_mm_encoder_outputs,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer

logger = init_logger(__name__)

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# 开启 ubatching 时，每一层会对应一个 metadata 列表；
# 否则直接是单个 metadata 字典。
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


# 为了支持“前向结束后，GPU->CPU 拷贝与后续工作重叠”而包装的输出对象。
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors: LogprobsTensors | None,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
        vocab_size: int,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # 记录拷贝流上的事件，后面 get_output() 会用它来等待异步拷贝完成。
        self.async_copy_ready_event = torch.Event()

        # 保留设备侧 tensor 的引用，避免在拷回 CPU 之前被提前释放。
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors

        # 在独立 stream 上发起异步拷贝，不在这里阻塞等待。
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking()
                if self._logprobs_tensors
                else None
            )
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """等待异步拷贝完成，并返回真正的 `ModelRunnerOutput`。"""
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        self.async_copy_ready_event.synchronize()

        # 拷贝完成后即可释放 GPU 侧引用，避免额外占显存。
        del self._logprobs_tensors
        del self._sampled_token_ids
        if max_gen_len == 1:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()
            logprobs_lists = None
            if self._logprobs_tensors_cpu is not None:
                logprobs_lists = self._logprobs_tensors_cpu.tolists()
        else:
            valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                self.sampled_token_ids_cpu,
                self.vocab_size,
                self._invalid_req_indices,
                logprobs_tensors=self._logprobs_tensors_cpu,
            )

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        output.logprobs = logprobs_lists
        return output


def _copy_pooler_output_to_cpu(
    raw_pooler_output: PoolerOutput, finished_mask: list[bool]
) -> list[torch.Tensor | None]:
    num_reqs = len(finished_mask)

    if isinstance(raw_pooler_output, torch.Tensor):
        if raw_pooler_output.shape[0] != num_reqs:
            raise ValueError(
                "Pooler output batch size does not match finished mask size: "
                f"{raw_pooler_output.shape[0]} != {num_reqs}."
            )

        num_finished = sum(finished_mask)
        if num_finished == 0:
            return [None] * num_reqs
        if num_finished == num_reqs:
            return list(raw_pooler_output.to("cpu", non_blocking=True))

        # 只有部分请求已经完成，只回传这些请求对应的 pooling 结果。
        finished_indices = [i for i, include in enumerate(finished_mask) if include]
        index_tensor = torch.tensor(
            finished_indices, device=raw_pooler_output.device, dtype=torch.long
        )
        finished_outputs = raw_pooler_output.index_select(0, index_tensor).to(
            "cpu", non_blocking=True
        )
        partial_pooler_output: list[torch.Tensor | None] = [None] * num_reqs
        for i, out in zip(finished_indices, finished_outputs):
            partial_pooler_output[i] = out
        return partial_pooler_output

    assert isinstance(raw_pooler_output, list)
    if len(raw_pooler_output) != num_reqs:
        raise ValueError(
            "Pooler output batch size does not match finished mask size: "
            f"{len(raw_pooler_output)} != {num_reqs}."
        )

    pooler_output: list[torch.Tensor | None] = [None] * num_reqs
    for i, (out, include) in enumerate(zip(raw_pooler_output, finished_mask)):
        if include and out is not None:
            pooler_output[i] = out.to("cpu", non_blocking=True)
    return pooler_output


class AsyncGPUPoolingModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        raw_pooler_output: PoolerOutput,
        finished_mask: list[bool],
        async_output_copy_stream: torch.cuda.Stream,
    ):
        self._model_runner_output = model_runner_output

        # 记录异步拷贝事件，供 get_output() 同步。
        self.async_copy_ready_event = torch.Event()

        # 保留 GPU 侧输出引用，直到 CPU 拷贝真正完成。
        self._raw_pooler_output = raw_pooler_output

        # 用独立 stream 异步拷贝 pooling 输出。
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self._model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
                raw_pooler_output=self._raw_pooler_output,
                finished_mask=finished_mask,
            )
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """等待 pooling 输出拷回 CPU，并返回最终输出。"""
        self.async_copy_ready_event.synchronize()

        # CPU 侧结果已经可用，释放设备侧引用。
        del self._raw_pooler_output
        return self._model_runner_output


class ExecuteModelState(NamedTuple):
    """`execute_model()` 与 `sample_tokens()` 之间临时传递的状态。

    当 `execute_model()` 只完成“前向 + logits 计算”，而真正的采样延后到
    `sample_tokens()` 执行时，就需要把这一轮的中间结果先暂存在这里。
    """

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None


class GPUModelRunner(
    LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin
):
    """v1 worker 侧真正驱动 GPU 执行的一层。

    可以把它理解成 scheduler 与模型前向之间的“翻译层”：

    1. 接收 scheduler 产出的本轮执行计划
    2. 把请求状态整理成 `InputBatch`
    3. 再把 `InputBatch` 转成模型前向需要的 GPU tensor / attention metadata
    4. 执行模型、采样 token、维护 KV cache / encoder cache / 持久批状态

    这个类很大，本质上围绕三条主线展开：
    1. 状态管理：请求从上一轮延续到下一轮时，哪些状态需要缓存
    2. 输入准备：如何把 scheduler 输出变成模型本轮的前向输入
    3. 执行优化：CUDA graph、异步拷贝、spec decode、KV sharing 等
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        # 保存常用配置，后续各执行路径会频繁使用。
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.offload_config = vllm_config.offload_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype

        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, self.model_config
        )

        self.is_pooling_model = model_config.runner_type == "pooling"
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model
        )
        # 真实值要等模型加载后才能知道，因此先给默认值。
        self.is_multimodal_pruning_enabled = False
        self.max_model_len = model_config.max_model_len

        # 若需要动态计算 KV scale，则只会在第一次前向前保留为 True。
        self.calculate_kv_scales = self.cache_config.calculate_kv_scales
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # external_launcher + PP 场景下，需要把最后一段的输出广播给其它 PP rank，
        # 这样各 rank 的状态推进才能保持一致。
        # TODO: 后续支持重叠执行的 micro-batches。
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and len(get_pp_group().ranks) > 1
        )

        # 与模型结构直接相关的派生信息。
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
        self.inputs_embeds_size = model_config.get_inputs_embeds_size()
        self.attention_chunk_size = model_config.attention_chunk_size
        # 仅 ALiBi 模型会用到。
        self.use_alibi = model_config.uses_alibi

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn
        self.is_mm_prefix_lm = self.model_config.is_mm_prefix_lm

        # 多模态能力相关开关与辅助对象。
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.uses_xdrope_dim = model_config.uses_xdrope_dim
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config
        )

        if self.model_config.is_encoder_decoder:
            # encoder-decoder 模型除了 decoder 长度，还要额外关心 encoder 输入上限。
            self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
        else:
            self.max_encoder_len = 0

        # 是否启用异步调度。启用后，一些 GPU->CPU 拷贝和记账工作会延后/并行处理。
        self.use_async_scheduling = self.scheduler_config.async_scheduling

        # 采样器负责把 logits 变成 token 结果。
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)

        self.eplb_state: EplbState | None = None
        # NOTE(yongji): 扩缩容期间临时关闭 EPLB 的开关。
        self.eep_eplb_suppressed = False
        """
        Expert parallelism 负载均衡器的运行状态。

        这里先声明，真正初始化在 `load_model()` 之后进行。
        """

        # 懒初始化成员：要等模型加载或 KV cache 初始化后才能真正就绪。
        # self.model: nn.Module  # 会在 `load_model()` 之后设置
        # 在 initialize_kv_cache 中填充。
        self.kv_caches: list[torch.Tensor] = []
        # 在 initialize_kv_cache_tensors 中填充。
        self.cross_layers_kv_cache: torch.Tensor | None = None
        self.cross_layers_attn_backend: type[AttentionBackend] | None = None
        # 结构为 [kv_cache_group_id][attn_group]。
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig，会在 KV cache 初始化阶段设置。

        # 多模态 encoder 输出缓存：mm_hash -> encoder_output。
        self.encoder_cache: dict[str, torch.Tensor] = {}

        self.use_aux_hidden_state_outputs = False
        # speculative decoding 相关对象。只有最后一个 PP rank 真正负责草稿生成。
        # NOTE(Jiayi): 当前把整个 draft model 都放在最后一个 PP rank 上，
        # 当草稿模型层数很多时，这并不是最理想的切分方式。
        if self.speculative_config and get_pp_group().is_last_rank:
            self.drafter: (
                NgramProposer  # noqa: F823
                | SuffixDecodingProposer
                | EagleProposer
                | DraftModelProposer
                | MedusaProposer
                | ExtractHiddenStatesProposer
            )
            if self.speculative_config.method == "ngram":
                from vllm.v1.spec_decode.ngram_proposer import NgramProposer

                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.uses_draft_model():
                self.drafter = DraftModelProposer(
                    vllm_config=self.vllm_config,
                    device=self.device,
                    runner=self,
                )
            elif self.speculative_config.method == "suffix":
                self.drafter = SuffixDecodingProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state
                    )
            elif self.speculative_config.method == "medusa":
                self.drafter = MedusaProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
            elif self.speculative_config.method == "extract_hidden_states":
                self.drafter = ExtractHiddenStatesProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
                self.use_aux_hidden_state_outputs = True
            else:
                raise ValueError(
                    "Unknown speculative decoding method: "
                    f"{self.speculative_config.method}"
                )
            self.rejection_sampler = RejectionSampler(self.sampler)

        self.num_spec_tokens = 0
        if self.speculative_config:
            self.num_spec_tokens = self.speculative_config.num_speculative_tokens
            draft_config = self.speculative_config.draft_model_config
            if draft_config is not None and draft_config.max_model_len is not None:
                self.effective_drafter_max_model_len = draft_config.max_model_len
            else:
                self.effective_drafter_max_model_len = self.max_model_len

        # 请求级缓存状态：保存跨 step 延续的请求信息。
        self.requests: dict[str, CachedRequestState] = {}
        # NOTE(rob): 这里只记录仍处于 prefill 阶段的请求的 prompt logprobs 需求。
        self.num_prompt_logprobs: dict[str, int] = {}
        self.comm_stream = torch.cuda.Stream()

        # 持久化批状态。
        # 它不是“这一轮临时创建一次”的 batch，而是跨轮复用、不断增删改的容器。
        # NOTE(Chen): 理想情况下，InputBatch 应该等 `initialize_kv_cache()` 拿到
        # 最终 kv cache 配置后再初始化；但受当前量化和权重卸载路径影响，
        # 若把它推迟到 `load_model()` 之后，某些场景会失败。
        # 因此这里先按默认 block size 初始化一次，后面若发现真实 block size 不同，
        # 再在 `initialize_kv_cache()` 中重建。
        logits_processors = model_config.logits_processors
        custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (
            tuple(logits_processors) if logits_processors is not None else ()
        )
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # encoder-decoder 模型要兼顾 cross-attention 的 KV cache 长度。
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
            kernel_block_sizes=[self.cache_config.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                custom_logitsprocs,
            ),
            # 自定义 logits processor 是否依赖历史输出 token 很难静态判断，
            # 这里保守地按“需要”处理。
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=self.is_pooling_model,
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
        )

        # 异步调度时，用单独的 stream 做 sampled token 的 GPU->CPU 拷贝。
        self.async_output_copy_stream: torch.cuda.Stream | None = None
        # 这个 event 用来保证“上一轮 DMA 还没读完 pinned 内存时，
        # 下一轮不会提前覆写它”。
        self.prepare_inputs_event: torch.Event | None = None
        if self.use_async_scheduling:
            self.async_output_copy_stream = torch.cuda.Stream()
            self.prepare_inputs_event = torch.Event()

        # 缓存设备属性，避免后续反复查询 CUDA 设备属性。
        self._init_device_properties()

        # 记录 encoder 耗时统计，供观测/指标上报使用。
        self.encoder_timing_registry: dict[str, EncoderTimingStats] = {}
        self._encoder_timing_lock = threading.Lock()

        # 为 CUDA graph 和高频执行路径预分配的持久 buffer。
        # 核心思想是：尽量避免每一轮都重新分配 tensor。
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.positions = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        self.encoder_seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        # inputs_embeds 可能是 bfloat16；这里不需要 numpy 视图，
        # 直接避免创建可规避相关报错。
        self.inputs_embeds = self._make_buffer(
            self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
        )
        self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
        self.discard_request_mask = self._make_buffer(
            self.max_num_reqs, dtype=torch.bool
        )
        self.num_decode_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int64
        )

        # 仅多模态模型使用。
        if self.supports_mm_inputs:
            # 双缓冲避免竞争：上一轮异步拷贝可能还在读 CPU buffer，
            # 当前轮已经开始往另一块 buffer 写入。
            self.is_mm_embed_buffers = [
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),
            ]
            self.is_mm_embed_idx = 0

        # 仅使用 M-RoPE 的模型会初始化这组位置张量。
        if self.uses_mrope:
            # NOTE: 这里故意多放一个 dummy 位置，使张量变成非连续布局，
            # 以便兼容 torch compile。
            # 详细背景见：https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: 打开 M-RoPE 后，position ids 总是 3 维。
            # 对纯文本输入来说，三个维度内容相同，因此效果等价于 1D-RoPE。
            # 论文解释见：https://arxiv.org/abs/2409.12191 第 5 页
            self.mrope_positions = self._make_buffer(
                (3, self.max_num_tokens + 1), dtype=torch.int64
            )

        # 仅使用 XD-RoPE 的模型会初始化这组位置张量。
        if self.uses_xdrope_dim > 0:
            # 类似 mrope，但维度数由模型指定，默认常见为 4。
            self.xdrope_positions = self._make_buffer(
                (self.uses_xdrope_dim, self.max_num_tokens + 1), dtype=torch.int64
            )

        # 第一段 PP rank 没有来自前一段的中间张量；其它 rank 在 load_model 后使用。
        self.intermediate_tensors: IntermediateTensors | None = None

        # 预先缓存一些常用 numpy 辅助数组，避免每一轮重复创建。
        # 用 int64 是为了长上下文下不溢出。
        self.arange_np = np.arange(
            max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens),
            dtype=np.int64,
        )

        # 跨层 KV sharing 的映射关系。
        # 若 `layer_name` 出现在这里，说明它会复用目标层的 KV cache。
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device
            )

        self.uniform_decode_query_len = 1 + self.num_spec_tokens

        # 运行时根据 batch 形状选择/调度 cudagraph 的分发器。
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        self.mm_budget = (
            MultiModalBudget(self.vllm_config, self.mm_registry)
            if self.supports_mm_inputs
            else None
        )

        self.reorder_batch_threshold: int | None = None

        # 这些 attention 层只存在于 runner 侧 KV cache 配置中，
        # scheduler 并不直接感知它们，例如 KV sharing 或 encoder-only attention。
        self.runner_only_attn_layers: set[str] = set()

        # 本轮/上一轮执行过程中缓存的输出。
        self._draft_token_ids: list[list[int]] | torch.Tensor | None = None
        self._draft_token_req_ids: list[str] | None = None
        self.transfer_event = torch.Event()
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )

        # 为 speculative decoding 预留的异步拷贝 buffer：
        # 一份拷“有效采样 token 数”，一份拷“draft token ids”。
        self.valid_sampled_token_count_event: torch.Event | None = None
        self.valid_sampled_token_count_copy_stream: torch.cuda.Stream | None = None
        # draft token 也可能被结构化输出等逻辑用到，所以同样支持异步拷回 CPU。
        self.draft_token_ids_event: torch.Event | None = None
        self.draft_token_ids_copy_stream: torch.cuda.Stream | None = None
        self.valid_sampled_token_count_cpu: torch.Tensor | None = None
        self.draft_token_ids_cpu: torch.Tensor | None = None
        if self.num_spec_tokens:
            self.draft_token_ids_event = torch.Event()
            self.draft_token_ids_copy_stream = torch.cuda.Stream()
            self.draft_token_ids_cpu = torch.empty(
                (self.max_num_reqs, self.num_spec_tokens),
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory,
            )
            if self.use_async_scheduling:
                self.valid_sampled_token_count_event = torch.Event()
                self.valid_sampled_token_count_copy_stream = torch.cuda.Stream()
                self.valid_sampled_token_count_cpu = torch.empty(
                    self.max_num_reqs,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )

        # 权重 offloader 需要尽早初始化，后面各处都会通过全局入口访问它。
        set_offloader(create_offloader(self.offload_config))

        # `execute_model()` 与 `sample_tokens()` 之间共享的临时状态。
        self.execute_model_state: ExecuteModelState | None = None
        self.kv_connector_output: KVConnectorOutput | None = None
        self.mamba_state_idx: dict[str, int] = {}
        self._mamba_copy_bufs: mamba_utils.MambaCopyBuffers | None = None
        self.layerwise_nvtx_hooks_registered = False

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        if self.speculative_config:
            draft_config = self.speculative_config.draft_model_config
            if draft_config is None or draft_config.max_model_len is None:
                self.effective_drafter_max_model_len = self.max_model_len

    def reset_mm_cache(self) -> None:
        """清理 profiling 阶段留下的多模态缓存。"""
        if self.mm_budget:
            self.mm_budget.reset_cache()

    def reset_encoder_cache(self) -> None:
        """清空 GPU 侧 encoder cache，避免复用旧权重算出的 embedding。"""
        self.encoder_cache.clear()

    @torch.inference_mode()
    def init_fp8_kv_scales(self) -> None:
        """从 sleep 唤醒后，重新恢复 FP8 KV cache 相关状态。

        核心做两件事：
        1. 把 KV cache tensor 清零，避免重新分配后的脏数据残留
        2. 把 attention 层里的 `_k_scale/_v_scale` 恢复到 1.0

        如果 scale 仍停留在唤醒后的默认 0.0，后续写入 KV cache 的值会被
        “压成 0”，最终输出通常会变成乱码。
        """
        if not self.cache_config.cache_dtype.startswith("fp8"):
            return

        kv_caches = getattr(self, "kv_caches", [])
        for cache_tensor in kv_caches:
            if cache_tensor is not None:
                cache_tensor.zero_()

        k_attr_names = ("_k_scale", "k_scale")
        v_attr_names = ("_v_scale", "v_scale")

        attn_layers = self.compilation_config.static_forward_context
        for name, module in attn_layers.items():
            if isinstance(module, (Attention, MLAAttention)):
                # TODO: 在线 FP8 KV cache 量化时，scale 通常就是 1.0；
                # 但有些压缩框架会离线校准出更好的 scale。未来这里可能需要恢复
                # 那些更精细的已校准值，而不只是简单写回 1.0。
                k_scale_val, v_scale_val = 1.0, 1.0

                # 恢复 K scale。
                for attr in k_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(k_scale_val)

                # 恢复 V scale。
                for attr in v_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(v_scale_val)

    def _get_positions(self, num_tokens: Any):
        if isinstance(num_tokens, int):
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, :num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, :num_tokens]
            return self.positions.gpu[:num_tokens]
        else:
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, num_tokens]
            return self.positions.gpu[num_tokens]

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=numpy,
        )

    def _get_mamba_copy_bufs(self) -> mamba_utils.MambaCopyBuffers:
        if self._mamba_copy_bufs is None:
            self._mamba_copy_bufs = mamba_utils.MambaCopyBuffers.create(
                self.max_num_reqs,
                self.kv_cache_config,
                self.model.get_mamba_state_copy_func(),
                self._make_buffer,
            )
        return self._mamba_copy_bufs

    def _init_model_kwargs(self):
        model_kwargs = dict[str, Any]()

        if not self.is_pooling_model:
            return model_kwargs

        num_reqs = self.input_batch.num_reqs
        pooling_params = self.input_batch.get_pooling_params()

        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if (
                param.extra_kwargs is not None
                and (token_types := param.extra_kwargs.get("compressed_token_type_ids"))
                is not None
            ):
                token_type_id_requests[i] = token_types

        if len(token_type_id_requests) == 0:
            return model_kwargs

        seq_lens = self.seq_lens.gpu[:num_reqs]
        token_type_ids = []

        for i in range(num_reqs):
            pos = token_type_id_requests.get(i, seq_lens[i])
            ids = (torch.arange(seq_lens[i]) >= pos).int()
            token_type_ids.append(ids)

        model_kwargs["token_type_ids"] = torch.concat(token_type_ids).to(
            device=self.device
        )
        return model_kwargs

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """必要时按 attention backend 的偏好重排 batch 中请求顺序。

        例如某些 backend（如 MLA）会希望把更偏计算密集或更偏访存密集的请求拆开，
        以获得更好的执行效率。
        """
        # 真正 attention-free 的模型其 kv_cache_groups 可能为 0；
        # 但像 Mamba 这类模型虽然不走 attention，也会用 kv cache 保存内部状态。
        # 因此这里只检查 kv_cache_groups 的数量，而不单看 is_attention_free。
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is not None:
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )

    # 供子类 / override 版本的 model runner 复用。
    def _init_device_properties(self) -> None:
        """初始化与设备属性相关的缓存字段。"""

        self.num_sms = num_compute_units(self.device.index)

    # 供子类 / override 版本的 model runner 复用。
    def _sync_device(self) -> None:
        torch.cuda.synchronize()

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """根据 `scheduler_output` 更新请求缓存和持久 batch。

        这是每轮执行最先发生的状态整理步骤，主要处理四类事：
        1. 删除已完成请求
        2. 把未被本轮调度的请求从持久 batch 中拿掉
        3. 把新请求 / 恢复执行的请求补进来
        4. 更新仍在运行请求的 token、block table、spec decode 状态

        这一步做完后，`self.input_batch` 就表示“本轮真正要送入模型”的批状态，
        `_prepare_inputs()` 会据此构造 GPU 输入张量。
        """
        # 1.先从缓存状态中移除已经结束的请求。
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)
        # 再把这些结束请求从持久 batch 中删除。
        # NOTE(woosuk): finished_req_ids 和 scheduled_req_ids 理论上可能重叠。
        # 例如某请求被 abort 后又用相同 req_id 重新提交。这里把它们视为
        # “旧请求结束 + 新请求开始”两件独立的事来处理。
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # 释放已经不再需要的多模态 encoder 输出缓存。
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # 2.从持久 batch 中移除“本轮没被调度”的请求。
        # NOTE(woosuk): 它们可能是被抢占了，也可能只是这一轮没轮到；
        # 因此这里只从 batch 里移走，不删除 requests 中的缓存状态。
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        # NOTE(zhuohan): 大多数情况下 cached_req_ids 与 resumed_req_ids 不相交。
        # 这里把 resumed 请求临时视作 unscheduled，是为了先清掉旧 batch 里的残留，
        # 后面再按“恢复请求”的正常路径重新加入。
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        # NOTE(woosuk): 持久 batch 优化默认假设相邻两轮请求集合高度重叠。
        # 如果两轮几乎完全不同，这种优化收益会明显下降。
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # 3.为本轮新增请求建立缓存状态。
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            if req_id in self.requests:
                # 流式会话可能复用已有 req_id，这时走增量更新路径。
                req_state = self._update_streaming_request(req_id, new_req_data)
                reqs_to_add.append(req_state)
                continue

            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

            # 只有用到 M-RoPE 的模型才需要为请求预计算这部分位置信息。
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            # 同理，XD-RoPE 也要在请求级别先准备好基础位置数据。
            if self.uses_xdrope_dim > 0:
                self._init_xdrope_positions(req_state)

            reqs_to_add.append(req_state)

        # 更新继续运行 / 从抢占中恢复的请求状态。
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        # 若上一轮启用了异步调度 + spec decode，这里可能要先等一份
        # GPU->CPU 异步拷贝完成，才能知道每个请求真实接受了多少 token。
        valid_sampled_token_count = self._get_valid_sampled_token_count()

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)

            if req_state.prev_num_draft_len and self.use_async_scheduling:
                # prev_num_draft_len 只在“异步调度 + speculative decoding”里有意义。
                # 它表示上一轮曾经生成过多少个 draft token，因此这一轮需要根据
                # 实际接受/拒绝数量，把 num_computed_tokens 校正回来。
                if req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    assert self.input_batch.prev_req_id_to_index is not None
                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    num_rejected = req_state.prev_num_draft_len - num_accepted
                    num_computed_tokens -= num_rejected
                    req_state.output_token_ids.extend([-1] * num_accepted)

            # 先更新 request 级缓存中的“已计算 token 数”。
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                if not req_data.new_token_ids:
                    # 异步调度 + PP：sampled token 会通过 GPU 广播传给其它 rank。
                    new_token_ids: list[int] = []
                else:
                    # 非异步 PP：由于首尾 stage 间没有直接通信，scheduler 会把
                    # sampled token ids 再带回前面的 stage。
                    new_token_ids = req_data.new_token_ids[i]
                    # 这些是上一轮真正确认过的输出 token，不包含尚未验收的 spec token。
                    num_new_tokens = (
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:
                        # 最常见情况是只新增 1 个 token，单独走快路径避免切片。
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        req_state.output_token_ids.extend(
                            new_token_ids[-num_new_tokens:]
                        )
            elif num_output_tokens < len(req_state.output_token_ids):
                # 若 sync-KV-load 失败，部分输出 token 可能被丢弃；
                # 这里把本地缓存状态裁回到一致状态。
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # 更新该请求对应的 block ids。
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # 正常继续执行时，只需把新分到的 block 追加上去。
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                # 抢占恢复后，会重新拿到一套 block 分配，直接整体替换。
                req_state.block_ids = new_block_ids

            if req_index is None:
                # 不在持久 batch 里，说明它要么刚恢复，要么上轮未入批，
                # 需要作为“重新加入 batch”的请求处理。

                if self.use_async_scheduling and num_output_tokens > 0:
                    # 异步调度下，恢复请求必须补回历史 output token ids，
                    # 否则下一轮构造 input_ids 会不正确。
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

                reqs_to_add.append(req_state)
                continue

            # 同步更新持久 batch 中与该请求相关的行。
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # 最后一个 PP rank 的 sampled token 已经在本地缓存，不需要再回写 token_ids_cpu。
            if not is_last_rank:
                # 把本轮新增 token 写回 InputBatch 的 CPU token buffer。
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index

            # 把 speculative decoding 的 draft token 也写回请求状态。
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)

        # 把新请求 / 恢复请求加入持久 batch。优先复用较小的空槽位。
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        # 清理删除请求留下的空洞，让 batch 更紧凑。
        self.input_batch.condense()
        # 某些 attention backend 可能希望按 decode/prefill 等重新排序 batch。
        self._may_reorder_batch(scheduler_output)
        # 最后统一刷新 batch metadata，供后续输入准备阶段使用。
        self.input_batch.refresh_metadata()

    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> None:
        """模型执行后，补更新 hybrid/spec decode 相关缓存状态。

        这主要服务于 MTP / EAGLE 等 hybrid 模型。
        对这类模型来说，draft token 的内部状态要先暂存下来，
        等确认每个请求最终接受了多少 token 后，下一轮再按接受数做对齐和搬移。
        """
        if not self.speculative_config or not self.model_config.is_hybrid:
            return

        # 统计每个请求最终接受了多少 token。
        num_reqs = output_token_ids.size(0)
        self.num_accepted_tokens.gpu[:num_reqs] = (
            (
                torch.cat(
                    [
                        output_token_ids,
                        torch.full(
                            (num_reqs, 1),
                            -1,
                            device=output_token_ids.device,
                        ),
                    ],
                    dim=1,
                )
                == -1
            )
            .int()
            .argmax(-1)
        )
        if self.cache_config.mamba_cache_mode == "align":
            for i, num_tokens in enumerate(
                self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()
            ):
                self.input_batch.num_accepted_tokens_cpu[i] = num_tokens

            mamba_utils.postprocess_mamba(
                scheduler_output,
                self.kv_cache_config,
                self.input_batch,
                self.requests,
                self.mamba_state_idx,
                self.compilation_config.static_forward_context,
                self.model.get_mamba_state_copy_func(),
                self._get_mamba_copy_bufs(),
            )
        else:
            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
            )

    def _update_streaming_request(
        self, req_id: str, new_req_data: NewRequestData
    ) -> CachedRequestState:
        """更新流式会话里的已有请求。

        这里会先把请求从 `InputBatch` 暂时移除，更新缓存状态，
        再让它以“重新入批”的方式参与后续流程。

        注意：
        `prompt_token_ids` 里可能已经包含之前生成出的中间输出 token。
        对流式场景来说，那些 token 现在已经变成新的输入上下文了。
        """
        self.input_batch.remove_request(req_id)
        req_state = self.requests[req_id]

        req_state.prompt_token_ids = new_req_data.prompt_token_ids
        req_state.mm_features = new_req_data.mm_features
        req_state.prompt_embeds = new_req_data.prompt_embeds
        req_state.sampling_params = new_req_data.sampling_params
        req_state.pooling_params = new_req_data.pooling_params
        req_state.block_ids = new_req_data.block_ids
        req_state.num_computed_tokens = new_req_data.num_computed_tokens
        req_state.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # 之前生成出来的 output token 已经并入 prompt_token_ids，
        # 因此这里清空旧的 output_token_ids。
        req_state.output_token_ids.clear()

        if self.uses_mrope:
            self._init_mrope_positions(req_state)

        return req_state

    def _init_mrope_positions(self, req_state: CachedRequestState):
        model = self.get_model()
        assert supports_mrope(model), "M-RoPE support is not implemented."
        assert req_state.prompt_token_ids is not None, (
            "M-RoPE requires prompt_token_ids to be available."
        )
        mrope_model = cast(SupportsMRoPE, model)

        req_state.mrope_positions, req_state.mrope_position_delta = (
            mrope_model.get_mrope_input_positions(
                req_state.prompt_token_ids,
                req_state.mm_features,
            )
        )

    def _init_xdrope_positions(self, req_state: CachedRequestState):
        model = self.get_model()
        xdrope_model = cast(SupportsXDRoPE, model)
        assert req_state.prompt_token_ids is not None, (
            "XD-RoPE requires prompt_token_ids to be available."
        )
        assert supports_xdrope(model), "XD-RoPE support is not implemented."

        req_state.xdrope_positions = xdrope_model.get_xdrope_input_positions(
            req_state.prompt_token_ids,
            req_state.mm_features,
        )

    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> BatchedTensorInputs:
        if not scheduler_output or not self.is_multimodal_raw_input_only_model:
            return {}

        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        for req in scheduler_output.scheduled_new_reqs:
            for feature in req.mm_features:
                if feature.data is not None:
                    mm_kwargs.append((feature.modality, feature.data))

        # 按 modality 聚合后一次性喂给模型。
        mm_kwargs_combined: BatchedTensorInputs = {}
        for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
        ):
            mm_kwargs_combined.update(mm_kwargs_group)

        return mm_kwargs_combined

    def _dummy_mm_kwargs(self, num_seqs: int) -> BatchedTensorInputs:
        if not self.is_multimodal_raw_input_only_model:
            return {}

        mm_budget = self.mm_budget
        assert mm_budget is not None

        if not mm_budget.mm_max_toks_per_item:
            return {}  # 没有需要 tower encoder 的模态，当前是纯 embedding 模式。

        dummy_modality = mm_budget.get_modality_with_max_tokens()
        return self._get_mm_dummy_batch(dummy_modality, num_seqs)

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回前缀和以及“按请求分段的局部 arange”。

        例如：
        `[2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])`

        等价于：
        `np.concatenate([np.arange(n) for n in num_tokens])`
        但这里的实现更快。
        """
        # 第 1 步：[2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # 第 2 步：[2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # 第 3 步：得到每个请求内部的局部 token 下标。
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _prepare_input_ids(
        self,
        scheduler_output: "SchedulerOutput",
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """为当前 batch 准备 `input_ids`。

        重点是处理 `prev_sampled_token_ids`：
        在异步调度下，上一轮刚采样出来的 token 可能还保存在 GPU 侧，
        这时需要把它们准确散射到本轮 `input_ids` 的对应位置里。
        """

        if self.input_batch.prev_sampled_token_ids is None:
            # 普通路径：直接把 CPU 侧准备好的 input_ids 拷到 GPU。
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        # 异步调度路径：上一轮 decode 出来的某些 token 还没写回 input_ids_cpu，
        # 需要直接从 GPU 上的 prev_sampled_token_ids 回填到本轮输入里。
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_common_req_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # 计算这些“上一轮和这一轮都存在的请求”在拍平 input_ids 中的落点。
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))
                total_num_spec_tokens += draft_len
                flattened_index = cu_num_tokens[cur_index].item() - 1
                # 例子：cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
                # 对应的 `sample_flattened_indices` = [0, 2, 5]
                # 对应的 `spec_flattened_indices` = [1,   3, 4,    6, 7]
                sample_flattened_indices.append(flattened_index - draft_len)
                spec_flattened_indices.extend(
                    range(flattened_index - draft_len + 1, flattened_index + 1)
                )
                start = prev_index * self.num_spec_tokens
                # prev_draft_token_indices 指出上一轮 draft_token_ids 中，
                # 哪些位置要被拷回到本轮 input_ids。
                # 例子：prev draft_tokens_id [[1,2], [3,4], [5,6]]
                # flatten 后为 [1,2,3,4,5,6]
                # 若各请求 draft_len 为 [1, 2, 1]
                # 则 prev_draft_token_indices = [0, 2, 3, 4]
                prev_draft_token_indices.extend(range(start, start + draft_len))
                indices_match &= prev_index == flattened_index
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_commmon_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        if num_commmon_tokens < total_without_spec:
            # 如果并不是所有请求都能从上一轮直接续上，就先把 CPU 侧 input_ids
            # 整体拷到 GPU，再局部覆盖那些可复用的 sampled tokens。
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_commmon_tokens == 0:
            # 与上一轮没有任何公共请求，那 CPU 侧 input_ids 已经是完整答案。
            return
        if indices_match and max_flattened_index == (num_commmon_tokens - 1):
            # 最常见优化路径：batch 完全没变，也没有重排，
            # 那就直接整段 copy，而不是走 scatter。
            self.input_ids.gpu[:num_commmon_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_commmon_tokens, 0],
                non_blocking=True,
            )
            if self.enable_prompt_embeds:
                self.is_token_ids.gpu[:num_commmon_tokens] = True
            return
        # 索引张量也异步上传到 GPU，这样 scatter 本身可以保持非阻塞。
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )

        # sampled token 填完后，再把 speculative draft tokens 也散射进去。
        if self._draft_token_ids is None or not spec_flattened_indices:
            return

        assert isinstance(self._draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)

        # input_ids 是 int32，因此 draft_token_ids 也要先转成 int32。
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)

        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

    def _get_encoder_seq_lens(
        self,
        num_scheduled_tokens: dict[str, int],
        kv_cache_spec: KVCacheSpec,
        num_reqs: int,
        for_cudagraph_capture: bool = False,
    ) -> tuple[torch.Tensor | None, np.ndarray | None]:
        if not isinstance(kv_cache_spec, CrossAttentionSpec):
            return None, None

        # 先把未真正参与本轮调度的 padding 槽位清零，避免复用上次捕获/执行的残值。
        self.encoder_seq_lens.np[:num_reqs] = 0

        # 构造 req_index -> encoder 输入长度 的映射，供 cross-attention 使用。
        for req_id in num_scheduled_tokens:
            req_index = self.input_batch.req_id_to_index[req_id]
            req_state = self.requests[req_id]
            if req_state.mm_features is None:
                self.encoder_seq_lens.np[req_index] = 0
                continue

            # 无论 encoder 本轮是否真的执行，这里都要告诉 decoder：
            # 这个请求理论上应当关注多少个 encoder token。
            encoder_input_tokens = sum(
                feature.mm_position.length for feature in req_state.mm_features
            )
            self.encoder_seq_lens.np[req_index] = encoder_input_tokens
        if for_cudagraph_capture:
            # CUDA graph 捕获时要用“足够真实”的 encoder 长度，
            # 否则捕获到的 max_seqlen_k 偏小，运行时可能选错 kernel。
            max_encoder_len = getattr(
                self.model_config.hf_config,
                "max_source_positions",
                self.max_encoder_len,
            )
            self.encoder_seq_lens.np[:num_reqs] = max_encoder_len

        self.encoder_seq_lens.copy_to_gpu(num_reqs)
        encoder_seq_lens = self.encoder_seq_lens.gpu[:num_reqs]
        encoder_seq_lens_cpu = self.encoder_seq_lens.np[:num_reqs]

        return encoder_seq_lens, encoder_seq_lens_cpu

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
    ]:
        """把 `InputBatch` 中的持久状态整理成本轮前向要用的输入。

        这一步的产物主要有两类：
        1. 真正送入模型的输入张量，如 `input_ids`、`positions`
        2. 采样/投机解码需要的辅助信息，如 `logits_indices`、spec metadata

        返回：
            `(logits_indices, spec_decode_metadata)`
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # 优化：先提交 block table 的拷贝，让它和后续 CPU 侧准备工作重叠。
        self.input_batch.block_table.commit_block_table(num_reqs)

        # 展开“每个 token 属于哪个请求”。
        # 例如 [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens 是前缀和，arange 是每个请求内部的局部 token 下标。
        # 例如：
        # 输入 [2, 5, 3] -> `cu_num_tokens=[2, 7, 10]`, `arange=[0, 1, 0, 1, 2, 3, 4, 0, 1, 2]`
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

        # 位置 = 历史已计算 token 数 + 本轮局部 offset。
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
            out=positions_np,
        )

        # M-RoPE 模型要额外生成 3D 位置编码。
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # XD-RoPE 模型同理。
        if self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)

        # 把二维 `[req_idx, token_pos]` 映射成拍平后的 token buffer 下标。
        # 例如：
        # [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # 对应拍平后下标 -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # 其中 M 是 max_model_len。
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)

        # NOTE(woosuk): 大张量场景下，torch.index_select 明显快于 np.take。
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
            )

        # InputBatch 并没有预分配一个巨大的 prompt_embeds CPU 缓冲区，
        # 因此这里要把各请求的 prompt embeds 手动拷入 runner 侧的大 buffer。
        if self.input_batch.req_prompt_embeds:
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # 该请求没有 embedding，跳过。
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # 本轮该请求没有调度 token，跳过。
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # 越过可用 embedding 范围时，不能再继续读取。
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # 仅拷贝当前这轮真正可用的那一段 embedding。
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[
                        output_idx : output_idx + actual_num_sched
                    ].copy_(req_embeds[start_pos:actual_end])

                output_idx += num_sched

        self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

        # 组装 attention metadata 依赖的基础张量。
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        # 注意：某些 kernel（如 FlashAttention）要求 query_start_loc 单调不减，
        # 因此 padding 段也要补成最后一个合法值。
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]

        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
        )
        # full cudagraph 模式下，未使用槽位也要有稳定值。
        self.seq_lens.np[num_reqs:].fill(0)
        self.seq_lens.copy_to_gpu()

        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # 标记哪些请求其实还没真正走到可采样位置。
        # 后面若误采样了这些请求，会在返回前清掉结果。
        self.discard_request_mask.np[:num_reqs] = (
            self.seq_lens.np[:num_reqs] < num_tokens_np
        )
        self.discard_request_mask.copy_to_gpu(num_reqs)

        # 最后把输入 token 相关张量准备到 GPU。
        self._prepare_input_ids(
            scheduler_output,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )

        if self.uses_mrope:
            # M-RoPE 的位置张量是 3D 的。
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            # XD-RoPE 也是多维位置张量。
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        else:
            # 普通文本模型最常见，直接复制 1D positions。
            self.positions.copy_to_gpu(total_num_scheduled_tokens)

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): chunked prefill 下，一个 batch 里可能包含“还没真正完成
            # 本轮可采样前向”的部分请求。这里为了简化实现仍会采样，
            # 但后面会忽略这些请求的 sampled token。
            # TODO: 后续补上 prompt logprobs 的支持。
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        else:
            # speculative decoding 时，需要知道每个请求本轮携带了多少个 draft token。
            # 只有少数请求可能带 draft，因此遍历字典会更高效。
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # chunked prefill 时，这里用 -1 作为 mask，避免 guided decoding
            # 在回滚 speculative token 时与“0 个 draft token”混淆。
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]
                ):
                    num_decode_draft_tokens[req_idx] = len(draft_token_ids)
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens
            )
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1
            # 某些 backend 的 decode-only cudagraph 会用到这份信息。
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()

        # 本轮若 LoRA 激活集合变化，这里同步切换到正确的活跃 LoRA。
        if self.lora_config:
            assert (
                np.sum(num_sampled_tokens)
                <= self.vllm_config.scheduler_config.max_num_batched_tokens
            )
            self.set_active_loras(
                self.input_batch, num_scheduled_tokens, num_sampled_tokens
            )

        return (
            logits_indices,
            spec_decode_metadata,
        )

    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        slot_mappings: dict[int, torch.Tensor] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """构造 attention backend 需要的 metadata。

        返回：
            `(attn_metadata, spec_decode_common_attn_metadata)`
        """
        # attention-free 模型不需要 attention metadata。
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None

        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        assert num_reqs_padded is not None and num_tokens_padded is not None

        attn_metadata: PerLayerAttnMetadata = {}
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]

        if for_cudagraph_capture:
            # 某些 backend（如 FA）在 sliding window 模型上，捕获时必须看到
            # 足够大的 max_seq_len，才能选中正确 kernel。
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.seq_lens.np[:num_reqs].max().item()

        if use_spec_decode:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            )
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        def _get_block_table(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                blk_table_tensor = torch.zeros(
                    (num_reqs_padded, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)

            # 未使用部分填 -1。full cudagraph 下 reshape_and_cache 需要这个约定；
            # 对 mamba 来说，-1 也和 PAD_SLOT_ID 对齐。
            blk_table_tensor[num_reqs:num_reqs_padded].fill_(-1)
            return blk_table_tensor

        assert slot_mappings is not None
        block_table_gid_0 = _get_block_table(0)
        slot_mapping_gid_0 = slot_mappings[0]

        if self.model_config.enable_return_routed_experts:
            self.slot_mapping = slot_mapping_gid_0[:num_tokens].cpu().numpy()
        cm_base = CommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens.gpu[:num_reqs_padded],
            _seq_lens_cpu=self.seq_lens.cpu[:num_reqs_padded],
            _num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[
                :num_reqs_padded
            ],
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            causal=True,
        )

        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
                self.seq_lens.cpu[:num_reqs],
                self.dcp_world_size,
                self.dcp_rank,
                self.parallel_config.cp_kv_cache_interleave_size,
            )
            self.dcp_local_seq_lens.cpu[num_reqs:].fill_(0)
            self.dcp_local_seq_lens.copy_to_gpu(num_reqs_padded)

            cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]
            cm_base.dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[
                :num_reqs_padded
            ]

        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices
            )

        # hybrid KV-cache group 之间如果 builder 类型和 KVCacheSpec 相同，
        # 往往只有 block table / slot mapping 不同。
        # 这里缓存第一次 build 的结果，后续尽量只做“替换 block table”的轻量更新。
        cached_attn_metadata: dict[
            tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
        ] = {}

        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
            ubid: int | None = None,
        ) -> None:
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]
            builder = attn_group.get_metadata_builder(ubid or 0)
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]
            cache_key = (kv_cache_spec, type(builder))

            cascade_attn_prefix_len = (
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
                if cascade_attn_prefix_lens
                else 0
            )

            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(
                builder, (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder)
            ):
                assert ubid is None, "UBatching not supported with GDN yet"
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                        :num_reqs_padded
                    ],
                )

            if for_cudagraph_capture:
                attn_metadata_i = builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            elif (
                cache_key in cached_attn_metadata
                and builder.supports_update_block_table
            ):
                attn_metadata_i = builder.update_block_table(
                    cached_attn_metadata[cache_key],
                    common_attn_metadata.block_table_tensor,
                    common_attn_metadata.slot_mapping,
                )
            else:
                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args,
                )
                if builder.supports_update_block_table:
                    cached_attn_metadata[cache_key] = attn_metadata_i

            if ubid is None:
                assert isinstance(attn_metadata, dict)
                attn_metadata_dict = attn_metadata
            else:
                assert isinstance(attn_metadata, list)
                attn_metadata_dict = attn_metadata[ubid]

            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = attn_metadata_i

        # 逐个 KV cache group 构造 metadata；
        # 同一 attention group 内的层会共享同一份 metadata，避免重复创建。
        spec_decode_common_attn_metadata = None
        for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
            cm = copy(cm_base)  # 浅拷贝即可，后面只改少量 group 相关字段。

            # 不同 kv_cache_group 间，通常只有 encoder_seq_lens、
            # block_table 和 slot_mapping 会变化。
            cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs_padded,
                for_cudagraph_capture=for_cudagraph_capture,
            )
            if kv_cache_gid > 0:
                cm.block_table_tensor = _get_block_table(kv_cache_gid)
                cm.slot_mapping = slot_mappings[kv_cache_gid]

            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, EagleProposer):
                    if self.drafter.kv_cache_gid == kv_cache_gid:
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm

            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                if ubatch_slices is not None:
                    for ubid, _cm in enumerate(split_attn_metadata(ubatch_slices, cm)):
                        _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)

                else:
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)

        if self.is_mm_prefix_lm:
            req_doc_ranges = {}
            for req_id in self.input_batch.req_ids:
                image_doc_ranges = []
                req_state = self.requests[req_id]
                for mm_feature in req_state.mm_features:
                    pos_info = mm_feature.mm_position
                    img_doc_range = pos_info.extract_embeds_range()
                    image_doc_ranges.extend(img_doc_range)
                req_idx = self.input_batch.req_id_to_index[req_id]
                req_doc_ranges[req_idx] = image_doc_ranges

            if isinstance(attn_metadata, list):
                for ub_metadata in attn_metadata:
                    for _metadata in ub_metadata.values():
                        _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]
            else:
                for _metadata in attn_metadata.values():
                    _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]

        if spec_decode_common_attn_metadata is not None and (
            num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
        ):
            # 目前 drafter 仍主要走 piecewise cudagraph，并且会原地修改
            # attention metadata，因此这里给它裁回未 padding 的视图。
            spec_decode_common_attn_metadata = (
                spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
            )

        return attn_metadata, spec_decode_common_attn_metadata

    def _compute_cascade_attn_prefix_lens(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: list[int],
    ) -> list[list[int]] | None:
        """计算各 KV cache group / attention group 的 cascade 前缀长度。

        返回值是二维列表 `cascade_attn_prefix_lens[kv_cache_group_id][attn_group_idx]`。
        如果本轮整体不适合启用 cascade attention，则返回 `None`。
        """

        use_cascade_attn = False
        num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        cascade_attn_prefix_lens: list[list[int]] = [
            [] for _ in range(num_kv_cache_groups)
        ]

        for kv_cache_gid in range(num_kv_cache_groups):
            for attn_group in self.attn_groups[kv_cache_gid]:
                if isinstance(attn_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                    cascade_attn_prefix_len = 0
                else:
                    # 返回 0 表示该 attention group 本轮不启用 cascade attention。
                    cascade_attn_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        num_computed_tokens,
                        num_common_prefix_blocks[kv_cache_gid],
                        attn_group.kv_cache_spec,
                        attn_group.get_metadata_builder(),
                    )
                cascade_attn_prefix_lens[kv_cache_gid].append(cascade_attn_prefix_len)
                use_cascade_attn |= cascade_attn_prefix_len > 0

        return cascade_attn_prefix_lens if use_cascade_attn else None

    def _compute_cascade_attn_prefix_len(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: int,
        kv_cache_spec: KVCacheSpec,
        attn_metadata_builder: AttentionMetadataBuilder,
    ) -> int:
        """计算 cascade attention 要使用的“公共前缀长度”。

        注意，这里返回的不是“请求真实共享了多少 token”，而是
        “当前实现下，允许 cascade attention 安全复用的那一段前缀长度”。

        当最终判断不启用 cascade attention 时，即使请求之间确实有公共前缀，
        这里也会返回 0。除此之外，返回值还会被裁到 block size 的整数倍，
        并受下面描述的实现约束继续收缩。

        参数：
            num_scheduled_tokens: 每个请求本轮调度的 token 数。
            num_common_prefix_blocks: 请求之间共享的 KV block 数量。

        返回：
            以 token 为单位的公共前缀长度。
        """

        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
        if common_prefix_len == 0:
            # 最常见情况：根本没有公共前缀。
            return 0

        # Cascade attention 会拆成两个 kernel：
        # 1. 先处理“公共前缀”
        # 2. 再处理剩余部分
        #
        # 第一个 kernel 会把所有请求的 query token 直接拼接起来，
        # 当作“同一个请求”的 query 去和公共前缀做双向注意力。
        # 这个阶段不会再做请求间 mask，因此公共前缀必须足够保守。
        #
        # 例子：
        # 请求 1 的输入 query: [D, E, X]
        # 请求 1 的 KV cache: [A, B, C, D, E, X]
        # 请求 1 的 num_computed_tokens: 3，也就是 [A, B, C]
        # 请求 2 的输入 query: [E, Y]
        # 请求 2 的 KV cache: [A, B, C, D, E, Y]
        # 请求 2 的 num_computed_tokens: 4，也就是 [A, B, C, D]
        #
        # 如果把 [A, B, C, D, E] 当成公共前缀，那么第一个 kernel 会让
        # 拼接后的 query [D, E, X, E, Y] 去双向关注 [A, B, C, D, E]。
        # 这会出错，因为请求 1 里的 D 不应该看到公共前缀里的 E；
        # 正确做法至少要把公共前缀截到 [A, B, C, D]。
        # 也就是说，公共前缀上界不能超过所有请求里最小的
        # `num_computed_tokens`，理论上再额外加 1 才能覆盖第一个 query token。
        #
        # 但当前实现为了始终使用“两段 kernel”流程，会进一步保守处理：
        # 最终只用 [A, B, C]，而不是 [A, B, C, D]。
        # 因为如果再多取一个 token，像下面这个请求 3：
        # 请求 3 的输入 query: [D]
        # 请求 3 的 KV cache: [A, B, C, D]
        # 请求 3 的 num_computed_tokens: 3，也就是 [A, B, C]
        # 那么请求 3 会被第一个 kernel 完全吃掉，第二个 kernel 反而拿到空输入；
        # 这虽然理论上可行，但当前实现并不支持。
        common_prefix_len = min(common_prefix_len, num_computed_tokens.min())
        # backend 的 cache 以 block 为单位管理，因此长度还要裁成 block 的整数倍。
        common_prefix_len = (
            common_prefix_len // kv_cache_spec.block_size * kv_cache_spec.block_size
        )
        use_sliding_window = isinstance(kv_cache_spec, SlidingWindowSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.sliding_window is not None
        )
        use_local_attention = isinstance(kv_cache_spec, ChunkedLocalAttentionSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.attention_chunk_size is not None
        )
        assert isinstance(kv_cache_spec, AttentionSpec)
        use_cascade = attn_metadata_builder.use_cascade_attention(
            common_prefix_len=common_prefix_len,
            query_lens=num_scheduled_tokens,
            num_query_heads=self.num_query_heads,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            use_alibi=self.use_alibi,
            use_sliding_window=use_sliding_window,
            use_local_attention=use_local_attention,
            num_sms=self.num_sms,
            dcp_world_size=self.dcp_world_size,
        )
        return common_prefix_len if use_cascade else 0

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt 段的位置已经在请求创建时预计算好了，直接切片复用。
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions.cpu[:, dst_start:dst_end] = req.mrope_positions[
                    :, src_start:src_end
                ]
                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # 新生成段的位置依赖当前上下文长度，只能按本轮状态现场推导。
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                assert req.mrope_position_delta is not None
                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions.np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _calc_xdrope_positions(self, scheduler_output: "SchedulerOutput"):
        xdrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.xdrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt 段的位置同样已经提前算好，直接搬到本轮大 buffer。
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.xdrope_positions.cpu[:, dst_start:dst_end] = req.xdrope_positions[
                    :, src_start:src_end
                ]
                xdrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # completion 段是新产生的 token，需要根据最新 context 动态续算。
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + completion_part_len

                XDRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.xdrope_positions.np,
                    out_offset=dst_start,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                xdrope_pos_ptr += completion_part_len

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # 这一步把“按请求组织的数据”转换成“拍平 token 视角”的索引。
        # 目标是让 target model 与 rejection sampler 都能高效定位：
        # 1. 哪些 logits 对应真实采样位置
        # 2. 哪些 logits 对应 draft token
        # 3. 哪些位置是 bonus token
        # 输入示例：
        # 示例中的 `cu_num_scheduled_tokens`: [  4, 104, 107, 207, 209]
        # 示例中的 `num_draft_tokens`:        [  3,   0,   2,   0,   1]
        # 输出示例：
        # 示例中的 `cu_num_draft_tokens`:     [  3,   3,   5,   5,   6]
        # 示例中的 `logits_indices`:          [  0,   1,   2,   3, 103, 104, 105, 106,
        #                                      206, 207, 208]
        # 示例中的 `target_logits_indices`:   [  0,   1,   2,   5,   6,   9]
        # 示例中的 `bonus_logits_indices`:    [  3,   4,   7,   8,  10]

        # 先算出“每个请求需要几个采样位置”：1 个真实 token + 若干 draft token。
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # 第 1 步：得到 sampled token 的前缀和与请求内局部下标。
        # 对应的 `cu_num_sampled_tokens`: [4, 5, 8, 9, 11]
        # 对应的 `arange`: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32
        )
        # 第 2 步：先把每个请求的 sampled token 映射到各自的起始 logits 下标。
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )
        # 第 3 步：再加上请求内 offset，得到最终 logits_indices。
        logits_indices += arange

        # bonus token 恰好是每个请求 sampled 段的最后一个位置。
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # 再构造“只针对 draft token”的那套索引。
        # 对应的 `cu_num_draft_tokens`: [3, 3, 5, 5, 6]
        # 对应的 `arange`: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: 这里仍有一次 CPU -> GPU 的索引传输，后续可以继续优化。
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True
        )
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(
            self.device, non_blocking=True
        )
        logits_indices = torch.from_numpy(logits_indices).to(
            self.device, non_blocking=True
        )
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True
        )
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True
        )

        # `draft_token_ids` 来自 input_ids 中“每个 target logits 的下一个 token”。
        # 对应的 `draft_token_indices`: [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def _prepare_kv_sharing_fast_prefill(
        self,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kv_sharing_fast_prefill_logits_indices is not None
        num_logits = logits_indices.shape[0]
        assert num_logits > 0
        self.kv_sharing_fast_prefill_logits_indices[:num_logits].copy_(logits_indices)
        # 预分配 buffer 可能残留上一轮更大的索引值；
        # 若当前 batch 更小，尾部脏值可能越界。
        # 因此把 padding 部分统一填成最后一个合法索引。
        self.kv_sharing_fast_prefill_logits_indices[num_logits:].fill_(
            logits_indices[-1].item()
        )
        # 这里只为 decoder 侧 dispatch，FULL graph 模式不在这条路径里处理。
        _, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_logits, invalid_modes={CUDAGraphMode.FULL}
        )
        num_logits_padded = batch_desc.num_tokens
        logits_indices_padded = self.kv_sharing_fast_prefill_logits_indices[
            :num_logits_padded
        ]
        return logits_indices_padded

    def _batch_mm_inputs_from_scheduler(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[
        list[str],
        list[tuple[str, MultiModalKwargsItem]],
        list[tuple[str, PlaceholderRange]],
    ]:
        """从 scheduler 指定的 encoder 输入里整理出多模态批数据。

        返回 `(mm_hashes, mm_kwargs, mm_lora_refs)`：
        - `mm_hashes`: 每个多模态条目的缓存键
        - `mm_kwargs`: 喂给多模态 encoder 的实际输入
        - `mm_lora_refs`: 每个条目对应的 `(req_id, placeholder_range)`，
          后续用于恢复它属于哪个请求、替换 prompt 中哪个区间
        """
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return [], [], []

        mm_hashes = list[str]()
        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        # 记录多模态条目对应的请求与占位区间，后面配置 tower/connector LoRA 会用到。
        mm_lora_refs = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue

                mm_hashes.append(mm_feature.identifier)
                mm_kwargs.append((mm_feature.modality, mm_feature.data))
                mm_lora_refs.append((req_id, mm_feature.mm_position))

        return mm_hashes, mm_kwargs, mm_lora_refs

    def _execute_mm_encoder(
        self, scheduler_output: "SchedulerOutput"
    ) -> list[torch.Tensor]:
        mm_hashes, mm_kwargs, mm_lora_refs = self._batch_mm_inputs_from_scheduler(
            scheduler_output
        )

        if not mm_kwargs:
            return []

        should_time = bool(
            self.observability_config
            and self.observability_config.enable_mm_processor_stats
            and scheduler_output.scheduled_encoder_inputs
        )

        # 尽量按 modality 做 batch，以提升 encoder 吞吐。
        # 但如果一个请求里混有多种 modality，或者顺序上需要保真，
        # 这里仍会拆成多个小组分别处理。
        # FIXME(ywang96): 当前做法是折中方案，真正彻底的解法应当是支持
        # encoder 输出重排，而不是靠这里手工维持顺序。
        model = cast(SupportsMultiModal, self.model)

        if self.lora_config and self.lora_manager.supports_tower_connector_lora():
            # encoder batch 的结构与主 decoder batch 不同，
            # 因此 tower/connector LoRA 的映射需要在这里单独重建。
            prompt_lora_mapping = []
            token_lora_mapping = []
            lora_requests = set()
            encoder_token_counts = []

            for req_id, pos_info in mm_lora_refs:
                req_idx = self.input_batch.req_id_to_index[req_id]
                lora_id = int(self.input_batch.request_lora_mapping[req_idx])

                # 优先使用 get_num_embeds，拿到更精确的多模态 embedding token 数。
                num_tokens = self.model.get_num_mm_encoder_tokens(  # type: ignore[attr-defined]
                    pos_info.get_num_embeds()
                )
                prompt_lora_mapping.append(lora_id)
                token_lora_mapping.extend([lora_id] * num_tokens)
                encoder_token_counts.append(num_tokens)

                if lora_id > 0:
                    lora_request = self.input_batch.lora_id_to_lora_request.get(lora_id)
                    if lora_request is not None:
                        lora_requests.add(lora_request)

            # 先设置 tower 侧 adapter 映射。
            tower_mapping = LoRAMapping(
                tuple(token_lora_mapping),
                tuple(prompt_lora_mapping),
                is_prefill=True,
                type=LoRAMappingType.TOWER,
            )
            self.lora_manager.set_active_adapters(lora_requests, tower_mapping)

            if hasattr(self.model, "get_num_mm_connector_tokens"):
                post_op_counts = [
                    self.model.get_num_mm_connector_tokens(num_tokens)  # type: ignore[attr-defined]
                    for num_tokens in encoder_token_counts
                ]

                connector_token_mapping = np.repeat(
                    np.array(prompt_lora_mapping, dtype=np.int32),
                    np.array(post_op_counts, dtype=np.int32),
                )
                connector_mapping = LoRAMapping(
                    index_mapping=tuple(connector_token_mapping.tolist()),
                    prompt_mapping=tuple(prompt_lora_mapping),
                    is_prefill=True,
                    type=LoRAMappingType.CONNECTOR,
                )

                self.lora_manager.set_active_adapters(
                    lora_requests,
                    connector_mapping,
                )

        encoder_outputs: list[torch.Tensor] = []
        # current_item_idx 用来把当前 modality 组重新对应回原始请求顺序。
        current_item_idx = 0
        for modality, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
        ):
            curr_group_outputs: MultiModalEmbeddings

            # EVS 相关的临时保护逻辑：
            # 当视频开启 pruning 时，scheduler 可能因为只看“裁剪后的视觉 token 数”
            # 而把太多样本塞进同一批，导致 encoder 峰值显存过高。
            # 这里临时退化成逐视频 micro-batch，优先避免 OOM。
            # TODO(ywang96): 等 memory profiling 正确计入 EVS 后再删除这段逻辑。
            if (
                self.is_multimodal_pruning_enabled
                and modality == "video"
                and num_items > 1
            ):
                curr_group_outputs_lst = list[torch.Tensor]()
                for video_idx in range(num_items):
                    video_mm_kwargs_item = mm_kwargs[current_item_idx + video_idx]
                    with self.timed_encoder_operation(
                        should_time, mm_lora_refs, current_item_idx + video_idx, 1
                    ):
                        _, _, micro_batch_mm_inputs = next(
                            group_mm_kwargs_by_modality(
                                [video_mm_kwargs_item],
                                device=self.device,
                                pin_memory=self.pin_memory,
                            )
                        )

                        micro_batch_outputs = model.embed_multimodal(
                            **micro_batch_mm_inputs
                        )

                        curr_group_outputs_lst.extend(micro_batch_outputs)

                curr_group_outputs = curr_group_outputs_lst
            else:
                # 正常路径：整组直接过 encoder。
                # `curr_group_outputs` 可能是：
                # 1. 一个形状为 `(num_items, feature_size, hidden_size)` 的张量，
                #    适用于各样本 feature_size 固定的场景；
                # 2. 一个长度为 `num_items` 的 list/tuple，
                #    每项是 `(feature_size, hidden_size)`，适用于动态长度场景。

                with self.timed_encoder_operation(
                    should_time, mm_lora_refs, current_item_idx, num_items
                ):
                    curr_group_outputs = model.embed_multimodal(**mm_kwargs_group)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )
            encoder_outputs.extend(curr_group_outputs)

            current_item_idx += num_items

        # encoder 输出按 mm_hash 缓存，后续 decoder 替换 embedding 时可直接复用。
        for mm_hash, output in zip(mm_hashes, encoder_outputs):
            self.encoder_cache[mm_hash] = output
            logger.debug("Finish execute for mm hash %s", mm_hash)
            self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)

        return encoder_outputs

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        # 双缓冲切换到另一块 CPU/GPU buffer，避免上一轮异步拷贝还没读完时发生竞争。
        self.is_mm_embed_idx = 1 - self.is_mm_embed_idx
        is_mm_embed_buf = self.is_mm_embed_buffers[self.is_mm_embed_idx]

        mm_embeds = list[torch.Tensor]()
        is_mm_embed = is_mm_embed_buf.cpu
        is_mm_embed[:total_num_scheduled_tokens] = False

        req_start_idx = 0
        should_sync_mrope_positions = False
        should_sync_xdrope_positions = False

        for req_id in self.input_batch.req_ids:
            mm_embeds_req: list[torch.Tensor] = []

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens

            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # 只有当“本轮要处理的 token 区间”和“该多模态特征覆盖区间”有重叠时，
                # 才需要把对应 encoder 输出取出来：
                # 区间一 `[num_computed_tokens, num_computed_tokens + num_scheduled_tokens)`
                # 与区间二 `[start_pos, start_pos + num_encoder_tokens)`
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # 后续特征起点已经越过本轮调度窗口，本轮不需要再看。
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # 这段 encoder 输出对应的 token 之前已经处理过，
                    # 其影响已经进入 decoder KV cache，无需重复注入。
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                # 当前重叠范围内如果没有真正的 embedding 槽位，就直接跳过。
                if curr_embeds_start == curr_embeds_end:
                    continue

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                # 多个 mm_feature 可能重叠，例如 video 里叠加 audio，
                # 因此这里对 mask 做按位或合并。
                if is_embed is None:
                    is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
                        True
                    )
                else:
                    is_mm_embed[
                        req_start_pos + start_idx : req_start_pos + end_idx
                    ] |= is_embed
                mm_embeds_req.append(mm_embeds_item)

            if self.is_multimodal_pruning_enabled and self.uses_mrope:
                assert req_state.mrope_positions is not None
                should_sync_mrope_positions = True
                mm_embeds_req, new_mrope_positions, new_delta = (
                    self.model.recompute_mrope_positions(
                        input_ids=req_state.prompt_token_ids,
                        multimodal_embeddings=mm_embeds_req,
                        mrope_positions=req_state.mrope_positions,
                        num_computed_tokens=req_state.num_computed_tokens,
                    )
                )
                req_state.mrope_positions.copy_(new_mrope_positions)
                req_state.mrope_position_delta = new_delta

            mm_embeds.extend(mm_embeds_req)
            req_start_idx += num_scheduled_tokens

        is_mm_embed = is_mm_embed_buf.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_mrope_positions:
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_xdrope_positions:
            self._calc_xdrope_positions(scheduler_output)
            self.xdrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        return mm_embeds, is_mm_embed

    def get_model(self) -> nn.Module:
        if not hasattr(self, "model"):
            raise ValueError("Cannot get model before model has been initialized")
        if isinstance(self.model, (CUDAGraphWrapper, UBatchWrapper)):
            # 执行阶段可能包着 cudagraph/ubatch wrapper，读取真实模型时要先解包。
            return self.model.unwrap()
        return self.model

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        if supports_realtime(model):
            supported_tasks.append("realtime")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        supported_tasks = list(model.pooler.get_supported_tasks())

        if "score" in supported_tasks:
            num_labels = getattr(self.model_config.hf_config, "num_labels", 0)
            if num_labels != 1:
                supported_tasks.remove("score")
                logger.debug_once("Score API is only enabled for num_labels == 1.")

        return supported_tasks

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        is_rs = is_residual_scattered_for_sp(self.vllm_config, num_tokens)

        # 开启 sequence parallelism 后，`residual` 会沿 TP 维切分；
        # 当前 rank 只需要同步并消费属于自己的那一段。
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                is_scattered = k == "residual" and is_rs
                copy_len = num_tokens // tp if is_scattered else num_tokens
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len], non_blocking=True
                )

        return IntermediateTensors(
            {
                k: v[: num_tokens // tp]
                if k == "residual" and is_rs
                else v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            }
        )

    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        """推进一次 EPLB（Expert Parallelism Load Balancing）状态机。"""
        if not self.parallel_config.enable_eplb or self.eep_eplb_suppressed:
            return

        assert self.eplb_state is not None
        model = self.get_model()
        assert is_mixture_of_experts(model)
        self.eplb_state.step(
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    def setup_eplb_from_mapping(
        self,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int,
    ) -> None:
        model = self.get_model()
        assert is_mixture_of_experts(model)

        self.eplb_state = EplbState.from_mapping(
            model=model,
            model_config=self.model_config,
            device=self.device,
            parallel_config=self.parallel_config,
            expanded_physical_to_logical=expanded_physical_to_logical,
            num_valid_physical_experts=old_num_physical_experts,
        )

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: KVConnectorOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        num_reqs = self.input_batch.num_reqs
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        hidden_states = hidden_states[:num_scheduled_tokens]
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]

        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np, seq_lens_cpu, device=hidden_states.device
        )

        model = cast(VllmModelForPooling, self.model)
        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )

        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            kv_connector_output=kv_connector_output,
        )

        if raw_pooler_output is None or not any(finished_mask):
            model_runner_output.pooler_output = [None] * num_reqs
            return model_runner_output

        if self.use_async_scheduling:
            return AsyncGPUPoolingModelRunnerOutput(
                model_runner_output=model_runner_output,
                raw_pooler_output=raw_pooler_output,
                finished_mask=finished_mask,
                async_output_copy_stream=self.async_output_copy_stream,
            )

        model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
            raw_pooler_output=raw_pooler_output,
            finished_mask=finished_mask,
        )
        self._sync_device()

        return model_runner_output

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        # SP 开启 collective fusion 时，token 数要补齐到 TP 大小的整数倍，
        # 这样各 rank 才能对齐做并行通信。
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if self.compilation_config.pass_config.enable_sp and tp_size > 1:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _prepare_mm_inputs(
        self, num_tokens: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.model.requires_raw_input_tokens:
            input_ids = self.input_ids.gpu[:num_tokens]
        else:
            input_ids = None

        inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
        return input_ids, inputs_embeds

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,  # Padded
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        is_first_rank = get_pp_group().is_first_rank
        is_encoder_decoder = self.model_config.is_encoder_decoder

        # `_prepare_inputs()` 可能重排 batch 顺序，
        # 所以多模态 embedding 必须在它之后再按最终顺序收集。
        ec_connector_output = None

        if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
            # 首个 PP rank 负责执行多模态 encoder，并把输出收集成本轮 decoder 输入。
            with self.maybe_get_ec_connector_output(
                scheduler_output,
                encoder_cache=self.encoder_cache,
            ) as ec_connector_output:
                self._execute_mm_encoder(scheduler_output)
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

            # 为了统一“普通 token”和“视觉/音频 embedding”两种输入形态，
            # 多模态模型这里始终走 embeddings 路径，即便某些位置其实只是文本 token。
            inputs_embeds_scheduled = self.model.embed_input_ids(
                self.input_ids.gpu[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            # TODO(woosuk): 这里仍有一次拷贝，可继续优化。
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)

            input_ids, inputs_embeds = self._prepare_mm_inputs(num_input_tokens)
            model_kwargs = {
                **self._init_model_kwargs(),
                **self._extract_mm_kwargs(scheduler_output),
            }
        elif self.enable_prompt_embeds and is_first_rank:
            # prompt embeds 模式下，一部分位置已经是 embedding，
            # 另一部分仍是普通 token id。
            # 这里先只把“仍是 token id 的位置”过 embedding 层，再回填到总 buffer。
            # TODO(qthequartermasterman): 长期看，embedding 层一直留在 graph 外并不理想。
            # v0 通过“分别为 input_ids / inputs_embeds 双编译 graph”规避了这个问题；
            # 若一个 batch 其实全是 token ids，那么把 embedding 层也纳入 cudagraph
            # 会更高效，效果接近下面 text-only 分支。
            token_ids_idx = (
                self.is_token_ids.gpu[:num_scheduled_tokens]
                .nonzero(as_tuple=False)
                .squeeze(1)
            )
            # 仅转换那些当前仍是 token id 的位置。
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs()
            input_ids = None
        else:
            # 纯文本模型直接走 token ids 路径即可。
            # 虽然也能像多模态一样预先转成 embeddings，但那会把 embedding 层排除在
            # cudagraph 之外，通常更慢。
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs()

        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # encoder-decoder 的 encoder 输入处理更简单：
            # encoder 输出直接作为 decoder 的 `encoder_outputs`，不需要做 prompt 替换。
            # 当前也只会有一个 encoder 输入。
            encoder_outputs = self._execute_mm_encoder(scheduler_output)
            model_kwargs.update({"encoder_outputs": encoder_outputs})

        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        )

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        # 采样阶段统一从 `input_batch.sampling_metadata` 读取各请求参数。
        sampling_metadata = self.input_batch.sampling_metadata
        # 异步调度下，如果本轮 sampling 参数依赖历史输出 token，
        # 这里要先把上一轮异步返回的 token 补进请求状态。
        self.input_batch.update_async_output_token_ids()
        if spec_decode_metadata is None:
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        # spec decode 且异步调度时，若 penalty / bad_words 等逻辑需要看到真实 draft token，
        # 这里先把上一轮的 draft token 回填到 sampling metadata。
        if self.use_async_scheduling and self._draft_token_req_ids is not None:
            draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
            self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)

        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        return sampler_output

    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> tuple[
        dict[str, int],
        LogprobsLists | None,
        list[list[int]],
        dict[str, LogprobsTensors | None],
        list[str],
        dict[str, int],
        list[int],
    ]:
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        num_reqs = self.input_batch.num_reqs
        discard_sampled_tokens_req_indices = np.nonzero(
            self.discard_request_mask.np[:num_reqs]
        )[0]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # 这些对象后面可能继续在 runner 内被复用/修改；
        # 先复制一份，避免异步调度下返回给上层后又被原地改动。
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        logprobs_lists = None
        if not self.use_async_scheduling:
            # 同步路径下，直接在当前 step 内把合法采样结果整理好。
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # 普通采样：每个请求最多只有 1 个真实输出 token。
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # 对 chunked prefill 等“不该返回采样结果”的请求做掩码清理。
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()

                if logprobs_tensors is not None:
                    logprobs_lists = logprobs_tensors.tolists()
            else:
                # spec decode：需要解析出接受/拒绝后的最终输出。
                valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    logprobs_tensors=logprobs_tensors,
                )
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            # 异步路径下，先把 sampled token 暂存在 GPU，避免本轮为了 CPU 同步而阻塞。
            # 下一轮 `_prepare_input_ids()` 会直接把它们散射回新的 input_ids。
            # spec decode 的类似工作则在 `propose_draft_token_ids()` 中完成。
            if self.input_batch.prev_sampled_token_ids is None:
                assert sampled_token_ids.shape[-1] == 1
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # 把最终采样 token 写回 runner 本地状态，
        # 这样下一轮 scheduler 不必再额外把它们带回来。
        # 例外是 PP 场景：由于首尾 stage 没有直接通信，scheduler 仍可能回传 token。
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # 若请求要求 prompt logprobs，这里顺手把本轮新增部分也计算出来。
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    @contextmanager
    def synchronize_input_prep(self):
        if self.prepare_inputs_event is None:
            yield
            return

        # 这批 pinned CPU tensor 会在多轮之间复用。
        # 异步调度下，上一次 H2D 传输可能还没结束，因此这里先显式等待。
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            self.prepare_inputs_event.record()

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        """对真实模型前向的一层薄封装。

        子类若想改执行方式，只需要覆写这里，而不必复制
        `execute_model()` 里那大段输入准备和后处理逻辑。

        参数：
            input_ids: 输入 token id
            positions: 位置编码张量
            intermediate_tensors: 上一 PP stage 传来的中间张量
            inputs_embeds: 输入 embedding，可替代 `input_ids`
            **model_kwargs: 其它模型前向参数

        返回：
            模型前向输出。
        """
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    @staticmethod
    def _is_uniform_decode(
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        force_uniform_decode: bool | None = None,
    ) -> bool:
        """判断是否为“所有请求 query 长度一致”的 uniform decode batch。"""
        return (
            (
                (max_num_scheduled_tokens == uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        # 仅用于 cudagraph 捕获。
        # TODO(lucas): model runner v2 应该重构这套捕获接口。
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
        )
        # encoder-decoder 只有在 decoder_step > 0、且当前不带 encoder 输出时，
        # 才适合走 cudagraph。并且它们不支持 chunked prefill，因此 batch 更接近 uniform。
        has_encoder_output = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # cudagraph 的 key 还要包含 LoRA 激活状态，因为不同活跃 LoRA 数
        # 可能对应不同的捕获图。
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora

        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)

        def dispatch_cudagraph(num_tokens, disable_full=False, valid_modes=None):
            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                num_active_loras=num_active_loras,
                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
            )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output
        )
        num_tokens_padded = batch_descriptor.num_tokens
        if self.compilation_config.pass_config.enable_sp:
            assert (
                batch_descriptor.num_tokens
                % self.vllm_config.parallel_config.tensor_parallel_size
                == 0
            ), (
                "Sequence parallelism requires num_tokens to be "
                "a multiple of tensor parallel size"
            )

        # DP 场景下，所有 rank 还需要额外协调 batch 形状，
        # 否则某些 rank 进图、某些 rank 不进图会导致执行分叉。
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.parallel_config,
                    allow_microbatching=allow_microbatching,
                    num_tokens_padded=num_tokens_padded,
                    uniform_decode=uniform_decode,
                    num_scheduled_tokens_per_request=num_scheduled_tokens_np,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )

            # 取出 DP 协调后的最终 token 数，并重新 dispatch 一次，
            # 保证 batch_descriptor 与各 rank 约定一致。
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # 重新 dispatch，拿到与 DP 补齐结果一致的 batch_descriptor。
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # 若不一致，说明 DP 协调结果与 runtime dispatch 已经失配。
                assert batch_descriptor.num_tokens == num_tokens_padded

        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )

        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _register_layerwise_nvtx_hooks(self) -> None:
        """按层注册 NVTX hook，用于细粒度追踪模型每一层/模块的耗时。"""

        if (
            self.vllm_config.observability_config.enable_layerwise_nvtx_tracing
            and not self.layerwise_nvtx_hooks_registered
        ):
            if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                logger.debug_once(
                    "layerwise NVTX tracing is not supported when CUDA graph is "
                    "turned off; you may observe part or all of the model "
                    "missing NVTX markers"
                )

            # STOCK_TORCH_COMPILE 模式下，如果在这里注册 hook，
            # `nn.Module.__call__` 会以 fullgraph=True 重新编译。
            # 但 nvtx.range_push/pop 不能被 torch dynamo 正确追踪，
            # hook 本身又会进入 tracing，因此这里只能跳过。
            if (
                self.vllm_config.compilation_config.mode
                == CompilationMode.STOCK_TORCH_COMPILE
            ):
                logger.debug_once(
                    "layerwise NVTX tracing is not supported when "
                    "CompilationMode is STOCK_TORCH_COMPILE, skipping "
                    "function hooks registration"
                )
            else:
                pyt_hooks = PytHooks()
                pyt_hooks.register_hooks(self.model, self.model.__class__.__name__)
                self.layerwise_nvtx_hooks_registered = True

    def _get_slot_mappings(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_tokens_unpadded: int,
        ubatch_slices: "UBatchSlices | None" = None,
    ) -> tuple[
        dict[int, torch.Tensor] | None,
        dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ]:
        """同时构造系统里会用到的两种 slot mapping 视图。

        参数：
            num_tokens_padded: padding 后的 token 总数
            num_reqs_padded: padding 后的请求总数
            num_tokens_unpadded: 实际未 padding 的 token 数
            ubatch_slices: DBO/ubatching 的切片信息

        返回：
            一个二元组：
            - `slot_mappings_by_gid`: attention metadata 使用的 `dict[int, Tensor]`
            - `slot_mappings_by_layer`: ForwardContext 使用的按层映射

        之所以要同时维护两种格式，是因为 attention backend 更关心
        “第几个 KV cache group 对应哪张 slot 表”，而前向上下文更常按层名索引。
        """
        if not (
            hasattr(self, "kv_cache_config")
            and self.kv_cache_config is not None
            and len(self.kv_cache_config.kv_cache_groups) > 0
        ):
            return None, None

        def _get_slot_mapping(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = self.kv_cache_config.kv_cache_groups[
                kv_cache_gid
            ].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]

            # 未使用部分统一填 -1。
            # full cudagraph 下 reshape_and_cache 依赖这个约定；
            # 对 mamba 来说，-1 也与 PAD_SLOT_ID 保持一致。
            slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)

            return slot_mapping

        slot_mappings_by_gid = {
            gid: _get_slot_mapping(gid)
            for gid, _ in enumerate(self.kv_cache_config.kv_cache_groups)
        }

        slot_mappings_by_layer: dict[str, torch.Tensor] = {}
        for gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            slot_mapping = slot_mappings_by_gid[gid]
            for layer_name in kv_cache_group.layer_names:
                slot_mappings_by_layer[layer_name] = slot_mapping

        if ubatch_slices is not None:
            result: list[dict[str, torch.Tensor]] = []
            for ubatch in ubatch_slices:
                sliced_mappings: dict[str, torch.Tensor] = {}
                for layer_name, slot_mapping in slot_mappings_by_layer.items():
                    sliced_mappings[layer_name] = slot_mapping[ubatch.token_slice]
                result.append(sliced_mappings)
            return slot_mappings_by_gid, result

        return slot_mappings_by_gid, slot_mappings_by_layer

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        """执行本轮模型前向，并把后续采样所需状态暂存下来。"""
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if self.vllm_config.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.clear_buffer()  # noqa
            else:
                logger.error("RoutedExpertsCapturer not initialized.")

        if scheduler_output.preempted_req_ids and has_kv_transfer_group():
            get_kv_transfer_group().handle_preemptions(
                scheduler_output.preempted_req_ids
            )

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            # 第 1 步：先把 scheduler 的决定同步到本地持久状态。
            self._update_states(scheduler_output)

            if has_ec_transfer() and get_ec_transfer().is_producer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend
                    == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    # 一个角落场景：external launcher + DP 时，外层循环可能仍认为
                    # “还有未结束请求”，但这一轮实际上没有 token 可执行。
                    # 这里先跑一次 dummy 路径，保证各 DP rank 的协调逻辑不失步。
                    self._dummy_run(1)
                if not has_kv_transfer_group():
                    # 本轮没有任何工作，直接返回空输出。
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            # 第 2 步：把持久 batch 整理成这一轮真正要送入模型的输入。
            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            cascade_attn_prefix_lens = None
            # ubatching 时暂不叠加 cascade attention。
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                # 预计算公共前缀长度，供 cascade attention 使用。
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            logger.debug(
                "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                "should_ubatch: %s, num_tokens_across_dp: %s",
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
            )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = (
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            )
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            logger.debug(
                "ubatch_slices: %s, ubatch_slices_padded: %s",
                ubatch_slices,
                ubatch_slices_padded,
            )

            # 若某个 backend 把 KV 更新拆到 forward 之外做，
            # slot_mapping 就要按 padding 后的维度构造。
            has_separate_kv_update = not all(
                all(
                    g.backend.forward_includes_kv_cache_update
                    for g in self.attn_groups[id]
                )
                for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
                if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
            )
            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            if self.cache_config.mamba_cache_mode == "align":
                mamba_utils.preprocess_mamba(
                    scheduler_output,
                    self.kv_cache_config,
                    self.cache_config,
                    self.mamba_state_idx,
                    self.input_batch,
                    self.requests,
                    self.compilation_config.static_forward_context,
                    self.model.get_mamba_state_copy_func(),
                    self._get_mamba_copy_bufs(),
                )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
                num_tokens_padded=num_tokens_padded
                if pad_attn or has_separate_kv_update
                else num_tokens_unpadded,
                num_reqs_padded=(
                    num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
                ),
                num_tokens_unpadded=num_tokens_unpadded,
                ubatch_slices=ubatch_slices_padded,
            )

            # 第 3 步：构建 attention backend 所需的 metadata。
            attn_metadata, spec_decode_common_attn_metadata = (
                self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded if pad_attn else None,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                    slot_mappings=slot_mappings_by_group,
                )
            )

            # 第 4 步：准备前向还需要的其它输入，例如多模态输入和 connector 数据。
            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors
            )

        # 动态计算 KV scale 会引入动态图操作，因此这一轮不能走 cudagraph。
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            # 首次前向后就视为 KV scale 已完成初始化。
            self.calculate_kv_scales = False

        # encoder-decoder 模型首次带 encoder 输入时不适合走编译路径。
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # 第 5 步：真正执行模型前向。
        # 若开启 spec decode，connector metadata 要延后到 draft 模型跑完后再清。
        clear_kv_metadata = self.speculative_config is None
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,
                skip_compiled=has_encoder_input,
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(
                scheduler_output, clear_metadata=clear_kv_metadata
            ) as kv_connector_output,
        ):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                # EAGLE3 会额外返回 auxiliary hidden states。
                hidden_states, aux_hidden_states = model_output
            else:
                # 常见情况只返回一份 hidden states。
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                # 常见路径：只有最后一个 PP rank 负责产出最终 logits。
                if not get_pp_group().is_last_rank:
                    # 非最后一个 PP stage 只需要向后传中间张量。
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # pooling 模型不做采样，直接返回 pooling 结果。
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        kv_connector_output,
                    )

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                # 少见路径：external launcher + PP，需要在 stage 间广播输出。
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_tokens_padded
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        # 生成模型会把“本轮 logits + 中间状态”缓存下来，交给 sample_tokens() 继续。
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        )
        self.kv_connector_output = kv_connector_output
        return None

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        """基于 `execute_model()` 暂存的结果完成采样、记账和输出组装。"""
        if self.execute_model_state is None:
            kv_connector_output = self.kv_connector_output
            self.kv_connector_output = None
            # 异步 PP 场景下，可能要先接收最后一个 PP rank 广播来的 token。
            if self.use_async_scheduling and get_pp_group().world_size > 1:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            if not kv_connector_output:
                return None  # type: ignore[return-value]

            # 只有 KV transfer 结果而没有前向输出时，直接透传 connector 输出。
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # 取出 execute_model() 暂存的中间状态。
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        ) = self.execute_model_state
        # 取出后立即清空，避免下一轮误用旧状态。
        self.execute_model_state = None

        # 若启用了结构化输出，先在 logits 上屏蔽非法 token。
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )
        if self.use_async_scheduling:
            pp = get_pp_group()
            # external launcher + broadcast_pp_output 时，logits 阶段已经同步过，
            # 这里无需再次广播 sampled token。
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(
                    sampler_output.sampled_token_ids
                )

        self._draft_token_ids = None
        self._draft_token_req_ids = None
        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                # 用本轮确认过的 token 继续提议下一轮 draft token。
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    slot_mappings,
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False
        if spec_config is not None:
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                <= self.effective_drafter_max_model_len
            )
            use_gpu_toks = (
                spec_config.use_eagle()
                or spec_config.uses_draft_model()
                or spec_config.uses_extract_hidden_states()
            ) and not spec_config.disable_padded_drafter_batch
            if use_gpu_toks:
                # EAGLE / DraftModel 类方案可以直接消费 GPU sampled tokens，
                # 因此不必等待 bookkeeping 结束。
                assert isinstance(
                    self.drafter,
                    EagleProposer | DraftModelProposer | ExtractHiddenStatesProposer,
                )
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count = (
                        self.drafter.prepare_next_token_ids_padded(
                            spec_decode_common_attn_metadata,
                            sampled_token_ids,
                            self.requests,
                            self.input_batch,
                            self.discard_request_mask.gpu,
                        )
                    )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count
                    )
                    # drafter 这一轮无法运行时，用全 0 draft token 占位。
                    self._draft_token_ids = torch.zeros(
                        1, device=self.device, dtype=torch.int32
                    ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
            else:
                propose_drafts_after_bookkeeping = input_fits_in_drafter

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
                spec_decode_metadata,
            )

        if propose_drafts_after_bookkeeping:
            # ngram 等方法依赖 CPU 侧 token，因此放到 bookkeeping 之后执行。
            propose_draft_token_ids(valid_sampled_token_ids)

        # spec decode 时要等 draft 模型也跑完，再清 connector metadata，
        # 否则草稿模型可能来不及保存自己的 KV cache。
        if self.speculative_config is not None:
            self.clear_kv_connector_metadata()

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        # drafting 过程可能会修改 kv_connector_output，这里统一取最终值。
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            if self.model_config.enable_return_routed_experts:
                capturer = RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capturer.save_captured_experts(indices=self.slot_mapping)  # noqa
                else:
                    logger.error("RoutedExpertsCapturer not initialized.")

            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
            )

        if not self.use_async_scheduling:
            return output

        with record_function_or_nullcontext(
            "gpu_model_runner: AsyncGPUModelRunnerOutput"
        ):
            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )
        with record_function_or_nullcontext(
            "gpu_model_runner: set_async_sampled_token_ids"
        ):
            # 若 batch 中有请求依赖历史输出 token，这里把异步拷回 CPU 的 tensor
            # 挂到 InputBatch 上，供下一轮输入准备直接复用。
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output

    def _pp_broadcast_prev_sampled_token_ids(
        self, sampled_token_ids: torch.Tensor
    ) -> None:
        """把最后一个 PP stage 采样出的 token 广播给其它 stage。"""
        pp = get_pp_group()
        assert pp.is_last_rank
        # `prev_sampled_token_ids` 约定形状为 [num_reqs, 1]。
        assert sampled_token_ids.dim() == 2 and sampled_token_ids.shape[-1] == 1, (
            "PP+async expects sampled_token_ids to have shape [num_reqs, 1]"
        )
        torch.distributed.broadcast(
            sampled_token_ids, src=pp.rank, group=pp.device_group
        )

    def _pp_receive_prev_sampled_token_ids_to_input_batch(self) -> None:
        """接收最后一个 PP stage 广播来的 sampled token，并写入 InputBatch。"""
        pp = get_pp_group()
        assert not pp.is_last_rank
        num_reqs = self.input_batch.num_reqs
        # `prev_sampled_token_ids` 约定形状为 [num_reqs, 1]。
        recv = torch.empty((num_reqs, 1), dtype=torch.int32, device=self.device)
        torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)
        self.input_batch.prev_sampled_token_ids = recv

        # 在这里构造 `prev_req_id_to_index`，供 `_prepare_input_ids()`
        # 把 req_id 映射回上一轮 batch 的行号。
        discard_req_indices = np.nonzero(self.discard_request_mask.np[:num_reqs])[0]
        discard_req_indices_set = set(discard_req_indices)
        prev_req_id_to_index: dict[str, int] = {}
        for i, req_id in enumerate(self.input_batch.req_ids):
            if i in discard_req_indices_set:
                continue
            prev_req_id_to_index[req_id] = i
            # PP+异步调度下，本地缓存的输出长度也要先前进一步，
            # 因此这里补一个占位 token（-1）。
            if (req_state := self.requests.get(req_id)) is not None:
                req_state.output_token_ids.append(-1)
        self.input_batch.prev_req_id_to_index = prev_req_id_to_index

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            return None
        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
        return DraftTokenIds(req_ids, draft_token_ids)

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False
    ) -> None:
        # 异步调度下，只有结构化输出、penalty、bad_words 等场景
        # 真正需要 CPU 侧 draft token 时才做拷贝。
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        # draft token 与 req_id 要成对保存，后面返回时才能对齐。
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        draft_token_ids: torch.Tensor = self._draft_token_ids
        if not torch.is_tensor(draft_token_ids):
            return
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_copy_stream is not None
        assert self.draft_token_ids_cpu is not None
        default_stream = torch.cuda.current_stream()
        num_reqs = draft_token_ids.shape[0]
        with torch.cuda.stream(self.draft_token_ids_copy_stream):
            if not zeros_only:
                # 异步把 draft token 拷回 CPU。
                self.draft_token_ids_copy_stream.wait_stream(default_stream)
                self.draft_token_ids_cpu[:num_reqs].copy_(
                    draft_token_ids, non_blocking=True
                )
            else:
                # 不需要真实拷贝时，直接把 CPU buffer 清 0 当占位。
                self.draft_token_ids_cpu[:num_reqs] = 0
            self.draft_token_ids_event.record()

    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:
        if isinstance(self._draft_token_ids, list):
            return self._draft_token_ids, self.input_batch.req_ids
        req_ids = self._draft_token_req_ids
        if req_ids is None:
            return [], []
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_cpu is not None
        self.draft_token_ids_event.synchronize()
        return self.draft_token_ids_cpu[: len(req_ids)].tolist(), req_ids

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        if self.valid_sampled_token_count_event is None:
            return

        default_stream = torch.cuda.current_stream()
        # 单独开一条 stream，把“有效采样 token 数”的拷回 CPU
        # 与下一轮 draft model 的输入准备重叠起来。
        with torch.cuda.stream(self.valid_sampled_token_count_copy_stream):
            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  # type: ignore
            counts = valid_sampled_tokens_count
            counts_cpu = self.valid_sampled_token_count_cpu
            assert counts_cpu is not None
            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)
            self.valid_sampled_token_count_event.record()

        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def _get_valid_sampled_token_count(self) -> list[int]:
        # 这里需要真正消费 CPU 结果，因此先等待异步拷贝完成。
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        sampled_count_event = self.valid_sampled_token_count_event
        if sampled_count_event is None or prev_sampled_token_ids is None:
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        assert counts_cpu is not None
        sampled_count_event.synchronize()
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ) -> list[list[int]] | torch.Tensor:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        spec_config = self.speculative_config
        assert spec_config is not None
        if spec_config.method == "ngram":
            from vllm.v1.spec_decode.ngram_proposer import NgramProposer

            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, NgramProposer)
            draft_token_ids = self.drafter.propose(
                sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                slot_mappings=slot_mappings,
            )
        elif spec_config.method == "suffix":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, SuffixDecodingProposer)
            draft_token_ids = self.drafter.propose(
                self.input_batch, sampled_token_ids, slot_mappings=slot_mappings
            )
        elif spec_config.method == "medusa":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, MedusaProposer)

            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                # 这种情况下 target model 的输入里不包含 draft token，
                # 采样点与请求一一对应，直接复用 sample_hidden_states。
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                assert spec_decode_metadata is not None, (
                    "No spec decode metadata for medusa"
                )
                for num_draft, tokens in zip(
                    spec_decode_metadata.num_draft_tokens, sampled_token_ids
                ):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1
                indices = torch.tensor(indices, device=self.device)
                hidden_states = sample_hidden_states[indices]

            draft_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
                slot_mappings=slot_mappings,
            )
        elif spec_config.uses_extract_hidden_states():
            assert isinstance(self.drafter, ExtractHiddenStatesProposer)
            assert isinstance(sampled_token_ids, torch.Tensor), (
                "sampled_token_ids should be a torch.Tensor for "
                "extract_hidden_states method."
            )
            if not self.use_aux_hidden_state_outputs or aux_hidden_states is None:
                raise ValueError(
                    "aux_hidden_states are required when using `extract_hidden_states`"
                )
            target_hidden_states = [h[:num_scheduled_tokens] for h in aux_hidden_states]

            draft_token_ids, drafter_kv_connector_output = self.drafter.propose(
                sampled_token_ids=sampled_token_ids,
                target_hidden_states=target_hidden_states,
                common_attn_metadata=common_attn_metadata,
                scheduler_output=scheduler_output,
                slot_mappings=slot_mappings,
            )
            # 合并两路 KVConnectorOutput；若只有一路非空，则直接保留那一路。
            # draft 路径和主模型路径都可能生成 connector 输出，这里统一合并。
            if self.kv_connector_output and drafter_kv_connector_output:
                self.kv_connector_output = KVConnectorOutput.merge(
                    self.kv_connector_output, drafter_kv_connector_output
                )
            else:
                self.kv_connector_output = (
                    self.kv_connector_output or drafter_kv_connector_output
                )

            next_token_ids, valid_sampled_tokens_count = (
                self.drafter.prepare_next_token_ids_padded(
                    common_attn_metadata,
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_mask.gpu,
                )
            )
            self._copy_valid_sampled_token_count(
                next_token_ids, valid_sampled_tokens_count
            )

        elif spec_config.use_eagle() or spec_config.uses_draft_model():
            assert isinstance(self.drafter, EagleProposer | DraftModelProposer)

            if spec_config.disable_padded_drafter_batch:
                # 不启用 padded drafter batch 时，drafter 直接消费 CPU 侧
                # `list[list[int]]`，其中无效请求对应空列表。
                assert isinstance(sampled_token_ids, list), (
                    "sampled_token_ids should be a python list when"
                    "padded-batch is disabled."
                )
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    scheduler_output.num_scheduled_tokens,
                )
            else:
                # 启用 padded batch 时，drafter 直接吃 GPU tensor，
                # 形状为 `(num_reqs, num_spec_tokens + 1)`；
                # 被拒绝的位置用 -1 占位。
                assert isinstance(sampled_token_ids, torch.Tensor), (
                    "sampled_token_ids should be a torch.Tensor when"
                    "padded-batch is enabled."
                )
                next_token_ids, valid_sampled_tokens_count = (
                    self.drafter.prepare_next_token_ids_padded(
                        common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_mask.gpu,
                    )
                )
                self._copy_valid_sampled_token_count(
                    next_token_ids, valid_sampled_tokens_count
                )

            num_rejected_tokens_gpu = None
            if spec_decode_metadata is None:
                token_indices_to_sample = None
                # 多模态模型里 input_ids 可能为空，但 drafter 这里只要 target 输入视图。
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                target_positions = self._get_positions(num_scheduled_tokens)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1
                    )
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                if spec_config.disable_padded_drafter_batch:
                    token_indices_to_sample = None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                        common_attn_metadata,
                        sampled_token_ids,
                        spec_decode_metadata.num_draft_tokens,
                    )
                    target_token_ids = self.input_ids.gpu[token_indices]
                    target_positions = self._get_positions(token_indices)
                    if self.use_aux_hidden_state_outputs:
                        assert aux_hidden_states is not None
                        target_hidden_states = torch.cat(
                            [h[token_indices] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[token_indices]
                else:
                    (
                        common_attn_metadata,
                        token_indices_to_sample,
                        num_rejected_tokens_gpu,
                    ) = self.drafter.prepare_inputs_padded(
                        common_attn_metadata,
                        spec_decode_metadata,
                        valid_sampled_tokens_count,
                    )
                    total_num_tokens = common_attn_metadata.num_actual_tokens
                    # padding 之后 token 已经被整理成连续区间，不再需要显式 gather 索引。
                    target_token_ids = self.input_ids.gpu[:total_num_tokens]
                    target_positions = self._get_positions(total_num_tokens)
                    if self.use_aux_hidden_state_outputs:
                        assert aux_hidden_states is not None
                        target_hidden_states = torch.cat(
                            [h[:total_num_tokens] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[:total_num_tokens]

            if self.supports_mm_inputs and self.drafter.supports_mm_inputs:
                mm_embed_inputs = self._gather_mm_embeddings(
                    scheduler_output,
                    shift_computed_tokens=1,
                )
            else:
                mm_embed_inputs = None

            draft_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
            )

        return draft_token_ids

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, (
                f"Config `{config_name}` not supported. "
                f"Allowed configs: {allowed_config_names}"
            )
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    @instrument(span_name="Loading (GPU)")
    def load_model(self, load_dummy_weights: bool = False) -> None:
        """加载模型权重，并完成与执行相关的后续包装。"""
        logger.info_once(
            "Starting to load model %s...",
            self.model_config.model,
            scope="global",
        )

        if self.parallel_config.enable_eplb:
            self.eplb_state = EplbState(self.parallel_config, self.device)
            eplb_models = 0

        try:
            with DeviceMemoryProfiler() as m:
                time_before_load = time.perf_counter()
                if load_dummy_weights:
                    self.load_config.load_format = "dummy"
                model_loader = get_model_loader(self.load_config)
                # 先加载主模型，再按需叠加 LoRA / drafter。
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config, model_config=self.model_config
                )
                if self.lora_config:
                    self.model = self.load_lora_model(
                        self.model, self.vllm_config, self.device
                    )
                if hasattr(self, "drafter"):
                    logger.info_once("Loading drafter model...")
                    self.drafter.load_model(self.model)
                    if (
                        hasattr(self.drafter, "model")
                        and is_mixture_of_experts(self.drafter.model)
                        and self.parallel_config.enable_eplb
                    ):
                        assert not self.parallel_config.enable_elastic_ep, (
                            "Elastic EP is not supported with drafter model."
                        )
                        spec_config = self.vllm_config.speculative_config
                        assert spec_config is not None
                        assert spec_config.draft_model_config is not None
                        logger.info_once(
                            "EPLB is enabled for drafter model %s.",
                            spec_config.draft_model_config.model,
                        )
                        if self.eplb_state is None:
                            self.eplb_state = EplbState(
                                self.parallel_config, self.device
                            )
                        self.eplb_state.add_model(
                            self.drafter.model,
                            spec_config.draft_model_config,
                        )
                        eplb_models += 1

                if self.use_aux_hidden_state_outputs:
                    if not supports_eagle3(self.get_model()):
                        raise RuntimeError(
                            "Model does not support EAGLE3 interface but "
                            "aux_hidden_state_outputs was requested"
                        )

                    # 优先使用 speculative config 指定的辅助层；
                    # 若未指定，再回退到模型默认设置。
                    aux_layers = self._get_eagle3_aux_layers_from_config()
                    if aux_layers:
                        logger.info(
                            "Using auxiliary layers from speculative config: %s",
                            aux_layers,
                        )
                    else:
                        aux_layers = self.model.get_eagle3_aux_hidden_state_layers()

                    self.model.set_aux_hidden_state_layers(aux_layers)
                time_after_load = time.perf_counter()
            self.model_memory_usage = m.consumed_memory
        except torch.cuda.OutOfMemoryError as e:
            msg = (
                "Failed to load model - not enough GPU memory. "
                "Try lowering --gpu-memory-utilization to free memory for weights, "
                "increasing --tensor-parallel-size, or using --quantization. "
                "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
                "for more tips."
            )
            combined_msg = f"{msg} (original error: {e})"
            logger.error(combined_msg)
            raise e
        logger.info_once(
            "Model loading took %s GiB memory and %.6f seconds",
            format_gib(self.model_memory_usage),
            time_after_load - time_before_load,
            scope="local",
        )
        if not load_dummy_weights:
            prepare_communication_buffer_for_model(self.model)
            if (drafter := getattr(self, "drafter", None)) and (
                drafter_model := getattr(drafter, "model", None)
            ):
                prepare_communication_buffer_for_model(drafter_model)
        mm_config = self.model_config.multimodal_config
        self.is_multimodal_pruning_enabled = (
            supports_multimodal_pruning(self.get_model())
            and mm_config is not None
            and mm_config.is_multimodal_pruning_enabled()
        )

        if (
            is_mixture_of_experts(self.model)
            and self.parallel_config.enable_eplb
            and not load_dummy_weights
        ):
            logger.info_once("EPLB is enabled for model %s.", self.model_config.model)
            assert self.eplb_state is not None
            self.eplb_state.add_model(
                self.model,
                self.model_config,
            )
            if self.eplb_state.is_async:
                self.eplb_state.start_async_loop()

        if (
            self.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
        ):
            backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
            compilation_counter.stock_torch_compile_count += 1
            self.model.compile(fullgraph=True, backend=backend)
            return
        # 其它编译模式下，cudagraph 的行为由 vLLM 自己的 wrapper/dispatcher 控制。

        # 如果需要 full cudagraph，这里把模型包上一层运行时封装。
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        if (
            cudagraph_mode.has_full_cudagraphs()
            and not self.parallel_config.use_ubatching
        ):
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )
        elif self.parallel_config.use_ubatching:
            if cudagraph_mode.has_full_cudagraphs():
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.FULL, self.device
                )
            else:
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.NONE, self.device
                )

        get_offloader().post_init()

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:
        """从 speculative config 中提取 Eagle3 的辅助层索引。

        这些索引决定 base model 的哪些 hidden states 会额外送给
        Eagle3 drafter，作为 speculative decoding 的辅助输入。

        返回：
            若 draft model config 里显式配置了层号，则返回对应元组；
            否则返回 `None`。
        """
        if not (self.speculative_config and self.speculative_config.draft_model_config):
            return None

        hf_config = self.speculative_config.draft_model_config.hf_config
        if not hasattr(hf_config, "eagle_aux_hidden_state_layer_ids"):
            return None

        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

        return None

    def reload_weights(
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
    ) -> None:
        """从给定迭代器或磁盘重新加载模型权重。

        参数：
            weights_iterator: 要写入模型的权重迭代器
            weights_path: 若未提供 `weights_iterator`，则从该路径加载；
                若也未提供，则回退到当前模型原始路径
            is_checkpoint_format: 若为 `False`，表示传入权重已经被转换成
                kernel format（例如已完成重排、重命名等）
        """
        # TODO(@kylesayrs): 这套逻辑未来应当抽到所有 runner / loader 共用。
        # 先做参数合法性校验。
        if weights_iterator is None and not is_checkpoint_format:
            logger.warning(
                "Reloading from disk means that weights will be in checkpoint format. "
                "Please use `is_checkpoint_format=True` "
                "to avoid weight reloading errors"
            )

        model = self.get_model()
        weights_to_load = {name for name, _ in model.named_parameters()}
        counter_before_reloading = time.perf_counter()

        # 未显式提供权重时，从磁盘重新读取。
        if weights_iterator is None:
            model_loader = get_model_loader(self.load_config)
            if not hasattr(model_loader, "get_all_weights"):
                raise NotImplementedError(
                    f"Model reloading with `{self.load_config.load_format}` format"
                )

            if weights_path is not None:
                self.model_config.model = weights_path
            weights_iterator = model_loader.get_all_weights(self.model_config, model)
            weights_iterator = cast(
                Iterable[tuple[str, torch.Tensor]], weights_iterator
            )

        # 真正开始把新权重写入当前模型。
        logger.info_once("Reloading weights inplace...", scope="local")
        load_device = (
            self.vllm_config.load_config.device or self.vllm_config.device_config.device
        )
        with torch.device(load_device):
            if is_checkpoint_format:
                # checkpoint/original format 需要走模型自己的 load_weights 逻辑，
                # 里面通常还会处理切分、重命名、后处理等步骤。
                initialize_layerwise_reload(model)
                loaded_weights = model.load_weights(weights_iterator)
                finalize_layerwise_reload(model, self.model_config)

            else:
                # kernel format 说明权重已经是目标布局，直接逐参数覆盖即可。
                logger.warning_once(
                    "Reloading with `is_checkpoint_format=True` requires that "
                    "weights be in kernel format and already sharded",
                    scope="local",
                )
                loaded_weights = set()
                for name, loaded_weight in weights_iterator:
                    param = model.get_parameter(name)  # TODO: buffers?
                    param.copy_(loaded_weight)
                    loaded_weights.add(name)

        # 记录耗时，并校验是否有参数未被加载。
        counter_after_reloading = time.perf_counter()
        diff_seconds = counter_after_reloading - counter_before_reloading
        logger.info_once(
            "Reloading and processing weights took %.2f seconds",
            diff_seconds,
            scope="local",
        )
        if self.model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                logger.warning(
                    "Following weights were not loaded from checkpoint: %s",
                    weights_not_loaded,
                )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # prompt logprobs 是低频功能，这里优先保证逻辑清晰、易维护，
        # 而不是为极致性能引入复杂实现。
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # 请求在 prefill 阶段被抢占时，本轮不会出现在调度结果里。
                continue

            # 取出该请求对应的 prompt 和状态信息。
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # prompt embeddings 模式下无法逐 token 计算 prompt logprobs。
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            # 准备承接结果的 LogprobsTensors。
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # 首次进入时，为整个 prompt 分配一份 CPU 侧结果缓冲区；
                # 若是 chunked prefill，后续会按切片逐段写入。
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            # 计算本轮实际需要取多少个 prompt logits。
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # 还只是中间 chunk，后面仍有 prompt token 尚未覆盖。
                # 注意等号场景下虽然本轮刚好算完，但仍延迟到下一步与生成 token 一起返回。
                num_logits = num_tokens
            else:
                # 最后一个 chunk，本轮之后就可以把整份 prompt logprobs 返回给上层。
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # 例如上一轮刚好已经 prefill 到 `num_prompt_tokens - 1`，
                # 那这轮就没有新的 prompt logprobs 需要产出。
                continue

            # 取出该请求在本轮对应的 hidden states，再单独算 logits。
            # 对 chunked prefill 来说，本轮覆盖了多少位置，就产出多少条 prompt logprob。
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # 第 i 个 prompt 位置对应的“目标 token”，其实是 prompt 中的第 i+1 个 token，
            # 因为我们要的是“看到前缀后，下一个 token 的概率”。
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # 计算并收集 top-k + 目标 token 的 logprob。
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # 结果异步写回 CPU 缓冲区，最后统一同步一次。
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        # 已经彻底结束 prefill 的请求，移出“待补全 prompt logprobs”集合。
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # 上面做的是 non-blocking GPU->CPU 拷贝，这里在真正返回前统一同步。
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: torch.Tensor | None,
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None and req_index < logits.shape[0]
                    else 0
                )
            return num_nans_in_logits
        except IndexError:
            return {}

    @contextmanager
    def maybe_randomize_inputs(
        self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None
    ):
        """按需随机化 dummy 输入，帮助 DP 场景下的专家选择更均衡。

        触发场景主要有两类：
        - `profile_run`
        - 某些 DP rank 在真实 batch 为空时执行 dummy run
        """

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        if not randomize_inputs:
            yield
        elif input_ids is not None:

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(
                    self.input_ids.gpu,
                    low=0,
                    high=self.model_config.get_vocab_size(),
                )

            logger.debug_once("Randomizing dummy input_ids for DP Rank")
            input_ids.copy_(rand_input_ids()[: input_ids.size(0)], non_blocking=True)
            yield
            input_ids.fill_(0)
        else:

            @functools.cache
            def rand_inputs_embeds() -> torch.Tensor:
                return torch.randn_like(
                    self.inputs_embeds.gpu,
                )

            assert inputs_embeds is not None
            logger.debug_once("Randomizing dummy inputs_embeds for DP Rank")
            inputs_embeds.copy_(
                rand_inputs_embeds()[: inputs_embeds.size(0)], non_blocking=True
            )
            yield
            inputs_embeds.fill_(0)

    def _get_mm_dummy_batch(
        self,
        modality: str,
        max_items_per_batch: int,
    ) -> BatchedTensorInputs:
        """生成多模态模型 profiling / 预编译使用的 dummy batch。"""
        assert self.mm_budget is not None

        # 这里只生成 1 个样本模板，再复制成 batch，避免重复做昂贵的 dummy 构造。
        dummy_mm_inputs = self.mm_registry.get_dummy_mm_inputs(
            self.model_config,
            mm_counts={modality: 1},
            cache=self.mm_budget.cache,
        )
        dummy_mm_item = dummy_mm_inputs["mm_kwargs"][modality][0]

        # 这里刻意让它“写入 cache 但不从 cache 读取”，
        # 这样后续真实 profiling 才能复用这份样本模板。
        assert dummy_mm_item is not None, "Item should not already be cached"

        return next(
            mm_kwargs_group
            for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                [(modality, dummy_mm_item)] * max_items_per_batch,
                device=self.device,
                pin_memory=self.pin_memory,
            )
        )

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """执行一次 dummy 前向，用于 warmup、profile 或 CUDA graph 捕获。

        参数：
            num_tokens: dummy 前向包含的 token 总数
            cudagraph_runtime_mode: 控制本次 dummy run 的 cudagraph 行为
                - `None`: 由 dispatcher 自动决定
                - `CUDAGraphMode.NONE`: 不进图，用于 warmup/profile
                - `CUDAGraphMode.PIECEWISE`: 分段图
                - `CUDAGraphMode.FULL`: 全图，此时需要完整 attention metadata
            force_attention: 即使不进 FULL graph，也强制构造 attention metadata；
                常用于预热 attention backend
            uniform_decode: 是否模拟 uniform decode batch
            skip_eplb: 是否跳过 EPLB 更新
            is_profile: 是否为 profile run
            create_mixed_batch: 是否构造“decode + prefill 混合批”
            remove_lora: 若为 `False`，dummy LoRA 在结束后不会立即销毁
            num_active_loras: 本次 dummy run 要模拟的活跃 LoRA 数量
        """
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # 当前 dummy_run 只覆盖 LM 主干，不覆盖纯 MM encoder-only 模型。
            return torch.tensor([]), torch.tensor([])

        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.is_valid_runtime_mode()
        )

        # 如果 decode_mode 是 FULL 且 separate_routine() 为真，
        # 说明 mixed prefill-decode 与 uniform decode 会走不同的图/例程。
        # uniform decode 指所有请求 query 长度一致，允许末尾有一个用于 padding 的短请求。
        # 这既可能是普通 decode（query_len == 1），也可能是 spec decode
        # （query_len == 1 + num_spec_decode_tokens）。
        #
        # 当 max_query_len == 1 时，FA2 会切到更适合纯 decode 的优化路径
        #（例如 Flashdecode 以及 GQA/MQA 优化）。
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # 根据 `num_tokens` 和 `max_num_seqs` 构造一份“请求分布”，
        # 让这批 dummy 请求合起来刚好覆盖目标 token 数。
        assert num_tokens <= self.max_num_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # 构造混合批：
            # 前半部分是若干个 decode 请求，最后拼一个 prefill 请求。
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # 先放若干个 1-token decode 请求，再放一个多 token 的 prefill 请求。
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # 注意：这里把 max_query_len 改成 prefill 长度，模拟 mixed batch 的上界。
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())

        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile
                or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                # `force_uniform_decode` 主要服务于 cudagraph 捕获。
                # mixed batch 捕获时有时也会满足 `num_tokens == num_reqs`，
                # 从外观上看像 uniform decode，但我们其实想捕 piecewise 图。
                force_uniform_decode=uniform_decode,
                # `force_has_lora` 也是为捕图准备。
                # 因为 LoRA 真正激活发生在更晚的上下文里，但 dispatch graph key 时
                # 就必须知道这次是否带 LoRA。
                force_has_lora=num_active_loras > 0,
                # `force_num_active_loras` 则用于捕获不同活跃 LoRA 数量对应的图。
                force_num_active_loras=num_active_loras,
            )
        )

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.vllm_config.parallel_config.num_ubatches,
        )
        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        attn_metadata: PerLayerAttnMetadata | None = None

        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=num_tokens,
            num_reqs_padded=num_reqs_padded,
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=ubatch_slices_padded,
        )

        # `_dummy_run()` 与 `execute_model()` 共享一批 pinned CPU buffer
        #（例如 `seq_lens`、`query_start_loc`）。
        # 因此它也必须走同一套 event 协议，避免前一轮 non_blocking H2D
        # 还在读时，后一轮 dummy/real run 提前覆写内存。
        with self.synchronize_input_prep():
            # `force_attention=True` 时无论是否 FULL graph 都构造 attention metadata；
            # 否则只有 FULL graph 才需要完整 attention 路径。
            if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
                if create_mixed_batch:
                    # mixed batch 的 warmup 里故意用更短的 seq_len，
                    # 以缩短预热耗时。
                    # TODO(luka): 需要更系统的 dummy batch 描述方式。
                    seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
                else:
                    seq_lens = max_query_len  # type: ignore[assignment]
                self.seq_lens.np[:num_reqs] = seq_lens
                self.seq_lens.np[num_reqs:] = 0
                self.seq_lens.copy_to_gpu()

                cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
                self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
                self.query_start_loc.copy_to_gpu()

                pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
                attn_metadata, _ = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs_padded,
                    max_query_len=max_query_len,
                    ubatch_slices=(ubatch_slices_padded if pad_attn else ubatch_slices),
                    for_cudagraph_capture=is_graph_capturing,
                    slot_mappings=slot_mappings_by_group,
                    use_spec_decode=self.speculative_config is not None,
                )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            remove_lora,
            num_active_loras,
        ):
            # dummy run 也必须遵守 runner 的总 token 上限。
            assert num_tokens_padded <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs()
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens_padded, None, False
                )

            if ubatch_slices_padded is not None:
                # 进入单个 ubatch 执行前，把若干全局值裁成“当前这个 ubatch”的视角。
                # TODO(sage,lucas): 这属于 padding 重构前遗留的兼容逻辑。
                num_tokens_padded = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    ubatch_slices=ubatch_slices_padded,
                    slot_mapping=slot_mappings,
                ),
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
            ):
                assert isinstance(
                    self.drafter,
                    EagleProposer | DraftModelProposer | ExtractHiddenStatesProposer,
                )
                assert self.speculative_config is not None
                # Eagle 目前只支持 PIECEWISE cudagraph。
                # 因此只有主模型也在兼容模式下时，drafter 才启用 cudagraph。
                # NOTE(lucas): 这是临时处理，后续需要清理。
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager

                # 当开启 `cudagraph_specialize_lora` 且存在活跃 LoRA 时，
                # 这里临时禁用一部分 cudagraph，规避 issue #28334。
                if (
                    self.compilation_config.cudagraph_specialize_lora
                    and num_active_loras > 0
                ):
                    use_cudagraphs = False

                self.drafter.dummy_run(
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    slot_mappings=slot_mappings,
                )

        # 把 layerwise NVTX hook 放到第一次 dynamo tracing 完成之后再注册，
        # 避免 hook 里的 NVTX 操作被 torch dynamo 追进图里，导致 graph break。
        # 对 DYNAMO_ONCE / VLLM_COMPILE 模式来说，编译模型的 tracing 通常只做一次，
        # 后续 `__call__` 会直接跳到编译结果，因此这里再注册 hook 是安全的。
        self._register_layerwise_nvtx_hooks()

        # DP 场景下，即使某些 rank 只是在跑 dummy batch，也可能需要同步推进 EPLB，
        # 否则各 rank 的专家重排节奏会失步。
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True
        )
        return hidden_states, hidden_states[logit_indices_device]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # dummy forward 产出的 hidden states 可能带有 `inf`/`nan` 等特殊值，
        # 为了不把 sampler 预热过程搞崩，这里换成一份随机张量。

        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # 纯 MM encoder-only 模型没有 sampler 路径。
            return torch.tensor([])

        hidden_states = torch.rand_like(hidden_states)

        logits = self.model.compute_logits(hidden_states)
        num_reqs = logits.size(0)

        dummy_tensors = lambda v: torch.full((num_reqs,), v, device=self.device)

        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            spec_token_ids=[[] for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        try:
            sampler_output = self.sampler(
                logits=logits, sampling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e
        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids, self.device
            )

            num_tokens = sum(len(ids) for ids in draft_token_ids)
            # 若以后需要模拟 `draft_probs`，可使用类似
            # `torch.randn(..., dtype=logits.dtype)` 这样的随机张量构造式。
            draft_probs = None
            logits = torch.randn(
                num_tokens + num_reqs,
                logits.shape[-1],
                device=self.device,
                dtype=logits.dtype,
            )
            self.rejection_sampler(
                dummy_spec_decode_metadata,
                draft_probs,
                logits,
                dummy_metadata,
            )
        return sampler_output

    def _dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_np = np.full(num_reqs, min_tokens_per_req)
        num_scheduled_tokens_np[-1] += num_tokens % num_reqs
        assert np.sum(num_scheduled_tokens_np) == num_tokens
        assert len(num_scheduled_tokens_np) == num_reqs

        req_num_tokens = num_tokens // num_reqs

        dummy_prompt_lens = torch.from_numpy(num_scheduled_tokens_np)
        dummy_token_ids = torch.zeros(
            (num_reqs, req_num_tokens), dtype=torch.int32, device=self.device
        )

        model = cast(VllmModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        dummy_pooling_params.verify(self.model_config)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            pooling_params=[dummy_pooling_params] * num_reqs,
            pooling_states=[PoolingStates() for i in range(num_reqs)],
        )

        dummy_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu=dummy_prompt_lens,
            device=hidden_states.device,
        )

        try:
            return model.pooler(
                hidden_states=hidden_states, pooling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler "
                    f"({task=}) with {num_reqs} dummy requests. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> PoolerOutput:
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # 纯 MM encoder-only 模型也不需要 pooler。
            return torch.tensor([])

        # 先找出输出体积最大的 pooling task，后续 warmup/profile 就按最重路径来做。
        supported_pooling_tasks = self.get_supported_pooling_tasks()

        if not supported_pooling_tasks:
            raise RuntimeError(
                f"Model {self.model_config.model} does not support "
                "any pooling tasks. See "
                "https://docs.vllm.ai/en/latest/models/pooling_models.html "
                "to learn more."
            )

        output_size = dict[PoolingTask, float]()
        for task in supported_pooling_tasks:
            # 每种 task 都用满批量跑一遍，确认不会在最坏情况下 OOM。
            output = self._dummy_pooler_run_task(hidden_states, task)
            output_size[task] = sum(o.nbytes for o in output if o is not None)
            del output  # 及时释放引用，方便后续 GC 回收。

        max_task = max(output_size.items(), key=lambda x: x[1])[0]
        return self._dummy_pooler_run_task(hidden_states, max_task)

    def profile_run(self) -> None:
        # 多模态模型先把 encoder 与 encoder cache 的显存占用也一并 profile 掉。
        if self.supports_mm_inputs:
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and mm_config.skip_mm_profiling:
                logger.info(
                    "Skipping memory profiling for multimodal encoder and "
                    "encoder cache."
                )
            else:
                mm_budget = self.mm_budget
                assert mm_budget is not None

                if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
                    if not mm_budget.mm_max_toks_per_item:
                        # 所有 modality 上限都为 0，说明是 embedding-only 模式。
                        # 虽然仍可能需要为 embedding 预留预算，但没有 encoder 可 profile。
                        logger.info(
                            "Skipping encoder profiling for embedding-only "
                            "mode (all modality limits=0 with "
                            "enable_mm_embeds=True).",
                        )
                    else:
                        # NOTE: 当前 profiling 只选“单个非文本 modality 且 token 最多”的情况，
                        # 即便模型实际支持多个 modality，也先按这条最重路径估算。
                        dummy_modality = mm_budget.get_modality_with_max_tokens()
                        max_mm_items_per_batch = mm_budget.mm_max_items_per_batch[
                            dummy_modality
                        ]

                        logger.info(
                            "Encoder cache will be initialized with a "
                            "budget of %s tokens, and profiled with "
                            "%s %s items of the maximum feature size.",
                            encoder_budget,
                            max_mm_items_per_batch,
                            dummy_modality,
                        )

                        # 构造多模态 dummy batch。
                        batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                            dummy_modality,
                            max_mm_items_per_batch,
                        )

                        # 真正执行一次 encoder，触发显存分配。
                        dummy_encoder_outputs = self.model.embed_multimodal(
                            **batched_dummy_mm_inputs
                        )

                        sanity_check_mm_encoder_outputs(
                            dummy_encoder_outputs,
                            expected_num_items=max_mm_items_per_batch,
                        )
                        for i, output in enumerate(dummy_encoder_outputs):
                            self.encoder_cache[f"tmp_{i}"] = output

        # `is_profile=True` 会顺手触发一些通信 buffer 的预分配。
        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True
        )
        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                output = self._dummy_pooler_run(hidden_states)
            else:
                output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None
        self._sync_device()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()

    @instrument(span_name="Capture model")
    def capture_model(self) -> int:
        """按预定 batch 形状捕获 CUDA graphs，并返回占用显存大小。"""
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()

        @contextmanager
        def freeze_gc():
            # 捕获期间尽量稳定对象图，减少 GC 对时序和内存的干扰。
            gc.collect()
            should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
            if should_freeze:
                gc.freeze()
            try:
                yield
            finally:
                if should_freeze:
                    gc.unfreeze()
                    gc.collect()

        # 按预先规划好的 batch 形状依次捕获。
        # 先捕获大形状，后续小形状可以尽量复用已分配的 memory pool。
        set_cudagraph_capturing_enabled(True)
        with freeze_gc(), graph_capture(device=self.device):
            start_free_gpu_memory = torch.cuda.mem_get_info()[0]

            for (
                runtime_mode,
                batch_descs,
            ) in self.cudagraph_dispatcher.get_capture_descs():
                self._capture_cudagraphs(
                    batch_descriptors=batch_descs,
                    cudagraph_runtime_mode=runtime_mode,
                )

            torch.cuda.synchronize()
            end_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # 捕获完成后全局关闭 cudagraph capturing。
        # 这样若后面还有意外捕获，会更容易被发现。
        set_cudagraph_capturing_enabled(False)

        # 锁住 workspace，避免执行阶段再发生尺寸扩张。
        lock_workspace()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # 这一步通常会比较慢，5~20 秒都不算异常。
        logger.info_once(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
            scope="local",
        )
        return cuda_graph_size

    def _capture_cudagraphs(
        self,
        batch_descriptors: list[BatchDescriptor],
        cudagraph_runtime_mode: CUDAGraphMode,
    ):
        assert (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and cudagraph_runtime_mode.is_valid_runtime_mode()
        ), f"Invalid cudagraph runtime mode: {cudagraph_runtime_mode}"

        if not batch_descriptors:
            return

        uniform_decode = batch_descriptors[0].uniform
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL

        dummy_run = functools.partial(
            self._dummy_run,
            uniform_decode=uniform_decode,
            skip_eplb=True,
            remove_lora=False,
            force_attention=force_attention,
        )

        # 只有全局 rank 0 打印捕获进度条，避免多卡日志互相打架。
        if is_global_first_rank():
            batch_descriptors = tqdm(
                batch_descriptors,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing CUDA graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    cudagraph_runtime_mode.name,
                ),
            )

        # 捕图阶段不推进 EPLB，避免把 dummy run 记成真实负载指标。
        for batch_desc in batch_descriptors:
            num_tokens = batch_desc.num_tokens
            num_active_loras = batch_desc.num_active_loras

            # 当前只有在以下条件同时满足时才捕获 ubatched graph：
            # 1. 使用 FULL cudagraph
            # 2. batch 属于 uniform decode
            # 3. token 数超过阈值
            # 否则退回普通非 ubatched 图。
            allow_microbatching = (
                self.parallel_config.use_ubatching
                and cudagraph_runtime_mode == CUDAGraphMode.FULL
                and uniform_decode
                and check_ubatch_thresholds(
                    config=self.vllm_config.parallel_config,
                    num_tokens=num_tokens,
                    uniform_decode=uniform_decode,
                )
            )

            for _ in range(self.compilation_config.cudagraph_num_of_warmups):
                # warmup 阶段统一用 `CUDAGraphMode.NONE`。
                # 但要注意，“是否 warm up attention”与“是否进图”是两回事：
                # FULL 意味着捕 attention，PIECEWISE 则不一定。

                dummy_run(
                    num_tokens,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    allow_microbatching=allow_microbatching,
                    num_active_loras=num_active_loras,
                )

            # 正式捕图。
            dummy_run(
                num_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                allow_microbatching=allow_microbatching,
                num_active_loras=num_active_loras,
                is_graph_capturing=True,
            )
        self.maybe_remove_all_loras(self.lora_config)

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """按 KV cache 分组初始化 attention backend。"""
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            layer_type = cast(type[Any], AttentionLayerBase)
            layers = get_layers_from_vllm_config(
                self.vllm_config, layer_type, kv_cache_group_spec.layer_names
            )
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # 用完整类名去重比直接用类对象更稳妥。
            # 某些动态生成的 backend 子类即便语义相同，也可能是不同对象。
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()

                if layer_name in self.kv_sharing_fast_prefill_eligible_layers:
                    attn_backend = create_fast_prefill_custom_backend(
                        "FastPrefill",
                        attn_backend,  # type: ignore[arg-type]
                    )

                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(
                    attn_backend, layer_kv_cache_spec
                )
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(
            attn_backends_map: dict[AttentionGroupKey, list[str]],
            kv_cache_group_id: int,
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend, kv_cache_spec), layer_names in attn_backends_map.items():
                attn_group = AttentionGroup(
                    attn_backend,
                    layer_names,
                    kv_cache_spec,
                    kv_cache_group_id,
                )

                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        # 在真正创建 metadata builder 之前，先确定 cudagraph_mode。
        self._check_and_update_cudagraph_mode(
            attention_backend_list, kv_cache_config.kv_cache_groups
        )

        # 顺带校验 attention backend 与 PCP / DCP 等并行特性的兼容性。
        check_attention_cp_compatibility(self.vllm_config)

        for i, attn_backend_map in enumerate(attention_backend_maps):
            self.attn_groups.append(create_attn_groups(attn_backend_map, i))

    def initialize_metadata_builders(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """为所有 KV cache group / attention group 创建 metadata builder。"""
        for kv_cache_group_id in range(len(kv_cache_config.kv_cache_groups)):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_sizes[kv_cache_group_id]
                    if kv_cache_group_id < len(kernel_block_sizes)
                    else None,
                    num_metadata_builders=1
                    if not self.parallel_config.use_ubatching
                    else self.parallel_config.num_ubatches,
                )
        # 某些 builder 会在初始化时修正 reorder 阈值，因此要放在它们之后计算。
        self.calculate_reorder_batch_threshold()

        # 如果有 drafter，也同步初始化它那边的 attention backend。
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
        ):
            assert isinstance(self.drafter, EagleProposer | DraftModelProposer)
            self.drafter.initialize_attn_backend(kv_cache_config, kernel_block_sizes)

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        """综合各 attention backend 的能力，决定最终的 cudagraph_mode。

        这里的本质是“求交集”：
        不同 backend 对 FULL / PIECEWISE / decode-only cudagraph 的支持能力不同，
        runner 需要取一个所有关键 backend 都能接受的模式。
        """
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_backend_name = None

        for attn_backend_set, kv_cache_group in zip(
            attention_backends, kv_cache_groups
        ):
            for attn_backend in attn_backend_set:
                builder_cls = attn_backend.get_builder_cls()

                cg_support = builder_cls.get_cudagraph_support(
                    self.vllm_config, kv_cache_group.kv_cache_spec
                )
                if cg_support.value < min_cg_support.value:
                    min_cg_support = cg_support
                    min_cg_backend_name = attn_backend.__name__
        # 根据 backend 的最弱能力，动态下调或修正用户配置的 cudagraph_mode。
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        # 先检查 mixed batch 的 full cudagraph 能否成立。
        if (
            cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL
            and min_cg_support != AttentionCGSupport.ALWAYS
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_backend_name} backend (support: "
                f"{min_cg_support})"
            )
            if min_cg_support == AttentionCGSupport.NEVER:
                # 完全不支持 full cudagraph，就只能直接报错。
                msg += (
                    "; please try cudagraph_mode=PIECEWISE, and "
                    "make sure compilation mode is VLLM_COMPILE"
                )
                raise ValueError(msg)

            # 否则尽量自动降级到还可工作的模式。
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_AND_PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_DECODE_ONLY
                )
            logger.warning(msg)

        # 再检查 decode full-cudagraph 是否受支持。
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_backend_name} backend (support: "
                f"{min_cg_support})"
            )
            if self.compilation_config.mode == CompilationMode.VLLM_COMPILE and (
                self.compilation_config.splitting_ops_contain_attention()
                or self.compilation_config.use_inductor_graph_partition
            ):
                msg += (
                    "; setting cudagraph_mode=PIECEWISE because "
                    "attention is compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += (
                    "; setting cudagraph_mode=NONE because "
                    "attention is not compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # speculative decoding 时，uniform decode 的 query 长度会变长，
        # 还要额外确认 decode full-cudagraph 仍然成立。
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and self.uniform_decode_query_len > 1
            and min_cg_support.value < AttentionCGSupport.UNIFORM_BATCH.value
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported"
                f" with spec-decode for attention backend "
                f"{min_cg_backend_name} (support: {min_cg_support})"
            )
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=NONE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # 自动降级后，再做一次兜底校验。
        if (
            cudagraph_mode.has_full_cudagraphs()
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            raise ValueError(
                f"CUDAGraphMode.{cudagraph_mode.name} is not "
                f"supported with {min_cg_backend_name} backend ("
                f"support:{min_cg_support}) "
                "; please try cudagraph_mode=PIECEWISE, "
                "and make sure compilation mode is VLLM_COMPILE"
            )

        # spec decode 下，uniform decode query 长度大于 1。
        # 若 decode graph 是独立捕获的，需要把 graph size 调整成这个长度的倍数。
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
            and self.uniform_decode_query_len > 1
        ):
            self.compilation_config.adjust_cudagraph_sizes_for_spec_decode(
                self.uniform_decode_query_len, self.parallel_config.tensor_parallel_size
            )

        # 模式确定后，再初始化 runtime dispatch 需要的 key。
        self.compilation_config.cudagraph_mode = cudagraph_mode
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, self.uniform_decode_query_len
        )

        # spec decode 的 drafter 也要拿到一致的 cudagraph key。
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(self.drafter, EagleProposer | ExtractHiddenStatesProposer)
            self.drafter.initialize_cudagraph_keys(cudagraph_mode)

    def calculate_reorder_batch_threshold(self) -> None:
        """从所有 attention group 中选一个最保守的 batch reorder 阈值。"""
        min_none_high = lambda a, b: a if b is None else b if a is None else min(a, b)

        reorder_batch_thresholds: list[int | None] = [
            group.get_metadata_builder().reorder_batch_threshold
            for group in self._attn_group_iterator()
        ]
        # 若模型根本没有 attention，或所有 backend 都没给阈值，就不启用重排。
        if len(reorder_batch_thresholds) == 0:
            self.reorder_batch_threshold = None
            return
        self.reorder_batch_threshold = reduce(min_none_high, reorder_batch_thresholds)  # type: ignore[assignment]

    def may_reinitialize_input_batch(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """当 block size 与初始假设不一致时，重建 `InputBatch`。"""
        block_sizes = []
        max_num_blocks = []
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            if isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            block_size = kv_cache_group.kv_cache_spec.block_size
            block_sizes.append(block_size)
            max_num_blocks_per_req = cdiv(
                max_model_len, block_size * get_total_cp_world_size()
            )
            if isinstance(kv_cache_group.kv_cache_spec, MambaSpec):
                max_num_blocks_per_req = (
                    max_num_blocks_per_req
                    if self.cache_config.enable_prefix_caching
                    else 1
                ) + kv_cache_group.kv_cache_spec.num_speculative_blocks
            max_num_blocks.append(max_num_blocks_per_req)

        if block_sizes != [self.cache_config.block_size] or kernel_block_sizes != [
            self.cache_config.block_size
        ]:
            assert self.offload_config.uva.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details."
            )
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                kernel_block_sizes=kernel_block_sizes,
                max_num_blocks_per_req=max_num_blocks,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                is_pooling_model=self.is_pooling_model,
            )

    def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """按配置先分配“原始 KV cache 内存块”，暂不调整最终视图。"""
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.zeros(
                kv_cache_tensor.size, dtype=torch.int8, device=self.device
            )
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors

    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:
        return itertools.chain.from_iterable(self.attn_groups)

    def _kv_cache_spec_attn_group_iterator(self) -> Iterator[AttentionGroup]:
        if not self.kv_cache_config.kv_cache_groups:
            return
        for attn_groups in self.attn_groups:
            yield from attn_groups

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        """把原始 KV cache 内存块重解释成各 backend 需要的 shape / dtype。"""
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            if group.kv_cache_group_id == len(kernel_block_sizes):
                # 末尾可能挂着一个“逻辑上参与分组、但实际上不分配 KV cache”的层组。
                continue
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    num_blocks_per_kv_block = (
                        kv_cache_spec.block_size // kernel_block_size
                    )
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block

                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        kernel_num_blocks,
                        kernel_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=self.cache_config.cache_dtype,
                    )
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                    # 不同 backend 对 stride/layout 约定不同。
                    # 这里先拿通用 shape，再按 backend 期望的 stride 顺序重排。
                    kv_cache_shape = tuple(
                        kv_cache_shape[i] for i in kv_cache_stride_order
                    )
                    # 最后再还原回逻辑上的原始维度视图。
                    inv_order = [
                        kv_cache_stride_order.index(i)
                        for i in range(len(kv_cache_stride_order))
                    ]
                    kv_caches[layer_name] = (
                        kv_cache_raw_tensors[layer_name]
                        .view(dtype)
                        .view(kv_cache_shape)
                        .permute(*inv_order)
                    )
                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    state_tensors = []
                    storage_offset_bytes = 0
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size
                        )
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size

                    kv_caches[layer_name] = state_tensors
                else:
                    raise NotImplementedError

        if has_attn and has_mamba:
            self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches

    def _update_hybrid_attention_mamba_layout(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """把 hybrid 布局中的 attention KV cache 从 `(2, num_blocks, ...)`
        调整成 `(num_blocks, 2, ...)`。

        参数：
            kv_caches: 各层绑定好的 KV cache buffer。
        """

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                kv_cache = kv_caches[layer_name]
                if isinstance(kv_cache_spec, AttentionSpec) and kv_cache.shape[0] == 2:
                    assert kv_cache.shape[1] != 2, (
                        "Fail to determine whether the layout is "
                        "(2, num_blocks, ...) or (num_blocks, 2, ...) for "
                        f"a tensor of shape {kv_cache.shape}"
                    )
                    hidden_size = kv_cache.shape[2:].numel()
                    kv_cache.as_strided_(
                        size=kv_cache.shape,
                        stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
                    )

    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        """初始化真正绑定到模型层上的 KV cache tensor。"""

        # 先尝试走针对 kv-connector 传输优化过的统一分配路径。
        cache_dtype = self.cache_config.cache_dtype
        if self.use_uniform_kv_cache(self.attn_groups, cache_dtype):
            kv_caches, cross_layers_kv_cache, attn_backend = (
                self.allocate_uniform_kv_caches(
                    kv_cache_config,
                    self.attn_groups,
                    cache_dtype,
                    self.device,
                    kernel_block_sizes,
                )
            )
            self.cross_layers_kv_cache = cross_layers_kv_cache
            self.cross_layers_attn_backend = attn_backend
        else:
            # 否则退回到通用路径：先分配原始 buffer，再重塑成目标布局。
            kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)

            # 把 raw buffer 变成各层真正可用的视图。
            kv_caches = self._reshape_kv_cache_tensors(
                kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes
            )

        # 设置跨层 KV sharing：让某些层直接复用目标层的 KV cache。
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
        return kv_caches

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        """把复用他人 KV cache 的层补进对应 KV cache group。"""
        if not self.shared_kv_cache_layers:
            # 没有跨层 KV sharing，就不需要额外处理。
            return

        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            self.runner_only_attn_layers,
        )

        if self.cache_config.kv_sharing_fast_prefill:
            # 在 You Only Cache Once 这类 KV sharing 方案里，prefill 阶段只需要
            # 真正产出 KV 的那些层，因此可以更早退出。
            attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
            for layer_name in reversed(attn_layers):
                if layer_name in self.shared_kv_cache_layers:
                    self.kv_sharing_fast_prefill_eligible_layers.add(layer_name)
                else:
                    break

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """按 `kv_cache_config` 完整初始化 KV cache 主线。

        这条主线依次做：
        1. 修正/补充 KV cache 配置
        2. 初始化 attention backend
        3. 计算 kernel block size 并创建 metadata builder
        4. 必要时重建 InputBatch
        5. 分配并绑定真正的 KV cache tensor
        6. 向 KV transfer / routed experts 等附属模块注册
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self._mamba_copy_bufs = None
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        # kernel block size 可能比 scheduler / kv_cache_manager 看到的 block_size 更小。
        # 例如调度侧按 256 token 一块管理，而 backend 只支持 64，那么这里会把
        # 一块逻辑 block 拆成 4 个 kernel block。
        kernel_block_sizes = prepare_kernel_block_sizes(
            kv_cache_config, self.attn_groups
        )

        # 创建运行时 attention metadata builder。
        self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

        # InputBatch 依赖最终 block size，因此必须在 attn backend 初始化后再决定是否重建。
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
        kv_caches = self.initialize_kv_cache_tensors(
            kv_cache_config, kernel_block_sizes
        )

        if (
            self.speculative_config
            and self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(self.drafter, ExtractHiddenStatesProposer)
            # 这类 drafter 要求相关层位于同一个 KV cache group。
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group():
            kv_transfer_group = get_kv_transfer_group()
            if self.cross_layers_kv_cache is not None:
                assert self.cross_layers_attn_backend is not None
                kv_transfer_group.register_cross_layers_kv_cache(
                    self.cross_layers_kv_cache, self.cross_layers_attn_backend
                )
            else:
                kv_transfer_group.register_kv_caches(kv_caches)
            kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    def init_routed_experts_capturer(self):
        logger.info(
            "Initializing routed experts capturer, enable_return_routed_experts: %s",
            self.model_config.enable_return_routed_experts,
        )
        routed_experts_capturer = RoutedExpertsCapturer.create()
        block_size = self.cache_config.block_size
        self.max_num_kv_tokens = (
            self.kv_cache_config.num_blocks // len(self.kv_cache_config.kv_cache_groups)
            + 1
        ) * block_size
        routed_experts_capturer.init_buffer(
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_kv_tokens=self.max_num_kv_tokens,
            vllm_config=self.vllm_config,
        )
        self._bind_routed_experts_capturer(routed_experts_capturer)

    def _bind_routed_experts_capturer(self, capturer: RoutedExpertsCapturer) -> None:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            BaseRouter,
        )

        for module in self.compilation_config.static_forward_context.values():
            if isinstance(module, FusedMoE) and isinstance(module.router, BaseRouter):
                layer_id = module.layer_id

                def _capture_fn(topk_ids, _layer_id=layer_id, _capturer=capturer):
                    _capturer.capture(_layer_id, topk_ids)

                module.router.set_capture_fn(_capture_fn)

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        """把 encoder-only attention 层补进 runner 侧 KV cache 配置。"""
        block_size = self.vllm_config.cache_config.block_size
        encoder_only_attn_specs: dict[AttentionSpec, list[str]] = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                attn_spec: AttentionSpec = EncoderOnlyAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                )
                encoder_only_attn_specs[attn_spec].append(layer_name)
                self.runner_only_attn_layers.add(layer_name)
        if len(encoder_only_attn_specs) > 0:
            assert len(encoder_only_attn_specs) == 1, (
                "Only support one encoder-only attention spec now"
            )
            spec, layer_names = encoder_only_attn_specs.popitem()
            self.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
            )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """从模型层上收集每个需要 KV cache 的层的 `KVCacheSpec`。"""
        if has_ec_transfer() and get_ec_transfer().is_producer:
            return {}
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention) and (
                kv_tgt_layer := attn_module.kv_sharing_target_layer_name
            ):
                # 这个层会直接复用目标层的 KV cache，因此这里故意不为它单独生成 spec。
                # 这样上层 KV cache 管理逻辑会把它视作“不需要独立分配缓存的层”，
                # 从而实现跨层 KV sharing 的节省内存效果。
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue
            # 不需要 KV cache 的模块（如 encoder-only attention）直接跳过。
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec

        return kv_cache_spec

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # 临时规避 issue #22754：
        # 直接 `tolist()` 会触发整条 CUDA 流同步，影响其它 stream 上的拷贝并发。
        # 这里先拷到 pinned CPU tensor，再做 tolist，代价更可控。
        pinned = self.sampled_token_ids_pinned_cpu[: sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return pinned.tolist()

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """取出并清空当前累计的 encoder 耗时统计。"""
        with self._encoder_timing_lock:
            stats = {
                req_id: stats_obj.to_dict()
                for req_id, stats_obj in self.encoder_timing_registry.items()
            }
            self.encoder_timing_registry.clear()
            return stats

    @contextmanager
    def timed_encoder_operation(
        self,
        should_time: bool,
        group_lora_refs: list[tuple[str, Any]],
        current_item_idx: int,
        num_items: int,
    ):
        """统计一组 encoder 前向的耗时。"""
        if not should_time:
            yield
            return

        group_refs = group_lora_refs[current_item_idx : current_item_idx + num_items]
        group_request_ids = {req_id for req_id, _ in group_refs}

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            per_request_time = elapsed / max(len(group_request_ids), 1)

            with self._encoder_timing_lock:
                for req_id in group_request_ids:
                    if req_id not in self.encoder_timing_registry:
                        self.encoder_timing_registry[req_id] = EncoderTimingStats()

                    stats = self.encoder_timing_registry[req_id]
                    stats.encoder_forward_secs += per_request_time
                    stats.num_encoder_calls += 1


@dataclass
class EncoderTimingStats:
    """单个请求的 encoder 前向耗时统计。"""

    encoder_forward_secs: float = 0.0
    """vision encoder 前向累计耗时，单位秒。"""

    num_encoder_calls: int = 0
    """该请求触发 encoder 的次数。"""

    def to_dict(self) -> dict[str, float | int]:
        return {
            "encoder_forward_secs": self.encoder_forward_secs,
            "num_encoder_calls": self.num_encoder_calls,
        }
