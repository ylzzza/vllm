# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

logger = init_logger(__name__)


@dataclass
class AttentionGroup:
    # 该 attention group 使用的 attention backend 类型，例如 FlashAttention 等。
    backend: type[AttentionBackend]
    # 属于这个 group 的 attention 层名列表。
    layer_names: list[str]
    # 这些层共享的 KV cache 规格，描述 block size、head size、dtype 等。
    kv_cache_spec: KVCacheSpec
    # 该 group 在 kv_cache_config.kv_cache_groups 中的下标。
    kv_cache_group_id: int
    # 开启 ubatching 时，每个 ubatch 都会有自己的 metadata builder。
    # 这样即使 builder 内部为 cudagraph 持有持久 buffer，也不会和其它 ubatch 冲突。
    metadata_builders: list[AttentionMetadataBuilder] = field(
        default_factory=lambda: []
    )

    def create_metadata_builders(
        self,
        vllm_config,
        device,
        kernel_block_size: int | None = None,
        num_metadata_builders: int = 1,
    ):
        # 有些 backend 实际 kernel 支持的 block size 小于 KV manager 管理的
        # block size；这里按 kernel_block_size 派生一份 builder 使用的 spec。
        kv_cache_spec_builder = (
            self.kv_cache_spec.copy_with_new_block_size(kernel_block_size)
            if kernel_block_size is not None
            else self.kv_cache_spec
        )
        # 每个 builder 后续负责把 batch 状态转换成该 backend 需要的
        # attention metadata。
        self.metadata_builders = [
            self.backend.get_builder_cls()(
                kv_cache_spec_builder,
                self.layer_names,
                vllm_config,
                device,
            )
            for _ in range(num_metadata_builders)
        ]

    def get_metadata_builder(self, ubatch_id: int = 0) -> AttentionMetadataBuilder:
        # 非 ubatching 场景默认只取第 0 个 builder；ubatching 时按 ubatch_id 取。
        assert len(self.metadata_builders) > ubatch_id
        return self.metadata_builders[ubatch_id]


def select_common_block_size(
    kv_manager_block_size: int, attn_groups: list[AttentionGroup]
) -> int:
    """
    选择一个所有 attention backend 都支持、且能整除 kv_manager_block_size
    的 kernel block size。

    如果 kv_manager_block_size 本身已经被所有 backend 支持，就直接返回它。
    否则，从各 backend 显式支持的整数 block size 中，选择最大的可行值。

    参数:
        kv_manager_block_size: KV cache manager 管理的逻辑 block size。
        attn_groups: 同一个 KV cache group 下的 attention groups。

    返回:
        选出的 kernel block size。

    异常:
        ValueError: 找不到所有 backend 都支持的 block size。
    """

    def block_size_is_supported(
        backends: list[type[AttentionBackend]], block_size: int
    ) -> bool:
        """检查给定 block size 是否被所有 backend 支持。"""
        for backend in backends:
            is_supported = False
            for supported_size in backend.get_supported_kernel_block_sizes():
                if isinstance(supported_size, int):
                    # backend 明确支持某个固定大小时，必须完全相等。
                    if block_size == supported_size:
                        is_supported = True
                elif isinstance(supported_size, MultipleOf):
                    # backend 声明支持某个基数的倍数时，只需要整除该基数。
                    if block_size % supported_size.base == 0:
                        is_supported = True
                else:
                    raise ValueError(f"Unknown supported size: {supported_size}")
            if not is_supported:
                return False
        return True

    backends = [group.backend for group in attn_groups]

    # 情况 1：KV cache manager 的逻辑 block size 已经被所有 backend 支持，
    # 直接使用它，不需要做虚拟 block 拆分。
    if block_size_is_supported(backends, kv_manager_block_size):
        return kv_manager_block_size

    # 情况 2：否则只能选一个更小的 kernel block size。
    # 这里收集所有 backend 显式声明支持的整数大小，从大到小尝试，返回第一个
    # 能整除 kv_manager_block_size 且被所有 backend 支持的大小。
    #
    # 为什么只需要枚举 int 格式的支持项：
    # 如果某个可行大小 b 对所有 backend 都只是满足 MultipleOf(x_i)，且 b 又能
    # 整除 kv_manager_block_size，那么 kv_manager_block_size 也会满足所有
    # MultipleOf(x_i)。这种情况已经会在“情况 1”直接返回 kv_manager_block_size。
    all_int_supported_sizes = set(
        supported_size
        for backend in backends
        for supported_size in backend.get_supported_kernel_block_sizes()
        if isinstance(supported_size, int)
    )

    for supported_size in sorted(all_int_supported_sizes, reverse=True):
        # kernel block size 必须能整除逻辑 block size，否则无法把一个逻辑 block
        # 干净地拆成若干 kernel block。
        if kv_manager_block_size % supported_size != 0:
            continue
        if block_size_is_supported(backends, supported_size):
            return supported_size
    raise ValueError(f"No common block size for {kv_manager_block_size}. ")


def prepare_kernel_block_sizes(
    kv_cache_config: KVCacheConfig, attn_groups: list[list[AttentionGroup]]
) -> list[int]:
    """
    为每个 KV cache group 生成实际 kernel 使用的 block size。

    对支持虚拟 block 拆分的 attention backend，会选择 backend 支持的大小。
    对 Mamba 这类非 attention cache，则直接使用原始 block size，不做拆分。

    参数:
        kv_cache_config: KV cache 配置。
        attn_groups: 按 KV cache group id 索引的 attention groups。

    返回:
        每个 cache group 对应的 kernel block size 列表。
    """
    kernel_block_sizes = []
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            # UniformTypeKVCacheSpecs 中所有层类型相同，因此随便取一层的 spec
            # 就足以判断它属于哪类 cache。
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            # encoder-only attention 的 cache 配置只服务 runner 侧 metadata，
            # 不参与普通 decoder KV cache 的 kernel block size 列表。
            continue
        if isinstance(kv_cache_spec, AttentionSpec):
            # attention cache 可能需要按 backend 能力把逻辑 block 拆成 kernel block。
            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, attn_groups[kv_cache_gid]
            )
            kernel_block_sizes.append(selected_kernel_size)
        elif isinstance(kv_cache_spec, MambaSpec):
            # Mamba 或其它非 attention cache 没有虚拟 block 拆分，直接沿用原大小。
            kernel_block_sizes.append(kv_cache_spec.block_size)
        else:
            raise NotImplementedError(
                f"unknown kv cache spec {kv_cache_group.kv_cache_spec}"
            )
    return kernel_block_sizes


def sanity_check_mm_encoder_outputs(
    mm_embeddings: MultiModalEmbeddings,
    expected_num_items: int,
) -> None:
    """
    对 [`vllm.model_executor.models.SupportsMultiModal.embed_multimodal`][]
    的返回结果做基本合法性检查。
    """
    # 多模态 encoder 输出必须是若干 2D embedding tensor，或者一个 3D tensor。
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    # 输出条数要和输入的多模态 item 数一致，否则后续无法按 item 对齐回请求。
    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    # 每个 item 的 embedding 期望是 [num_tokens, hidden_size]。
    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )


def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """
    根据 gpu_memory_utilization 计算 vLLM 计划使用的显存量，并检查当前空闲显存
    是否足够。
    """
    # requested_memory 是 vLLM 希望占用的总显存上限，而不是当前已用显存。
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )

    # 启动时空闲显存不足会直接报错，避免后面加载权重或初始化 KV cache 时 OOM。
    if init_snapshot.free_memory < requested_memory:
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            f"({format_gib(init_snapshot.free_memory)}/"
            f"{format_gib(init_snapshot.total_memory)} GiB) on startup "
            f"is less than desired GPU memory utilization "
            f"({cache_config.gpu_memory_utilization}, "
            f"{format_gib(requested_memory)} GiB). Decrease GPU memory "
            f"utilization or reduce GPU memory used by other processes."
        )

    return requested_memory


def add_kv_sharing_layers_to_kv_cache_groups(
    shared_kv_cache_layers: dict[str, str],
    kv_cache_groups: list[KVCacheGroupSpec],
    runner_only_attn_layers: set[str] | None = None,
) -> None:
    """
    根据 `shared_kv_cache_layers` 配置跨层 KV cache sharing。

    某些 attention 层不会为自己单独分配 KV cache，而是复用目标层已经分配好的
    KV cache。本函数负责把这些“复用者层”补进目标层所在的 KV cache group。
    这样后续构建 attention metadata 时，这些层仍然能拿到对应 metadata。

    参数:
        shared_kv_cache_layers: 跨层 KV cache sharing 的层映射关系。
            如果某个 Attention 层 `layer_name` 出现在这个 dict 的 key 中，
            表示它执行 attention 时会使用
            `shared_kv_cache_layers[layer_name]` 这个目标层 KV cache 里的
            keys 和 values。
        kv_cache_groups: 模型的 KV cache groups。
        runner_only_attn_layers: 只在 runner 侧补入的 attention 层集合。
    """
    layer_to_kv_cache_group: dict[str, KVCacheGroupSpec] = {}
    for kv_cache_group in kv_cache_groups:
        for layer_name in kv_cache_group.layer_names:
            # 先建立目标层到其 KV cache group 的反向索引。
            layer_to_kv_cache_group[layer_name] = kv_cache_group

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        # 找到目标层所在 group，再把复用者层追加进去。
        # 注意：这里没有分配新 cache，只是让 metadata 构建逻辑也看见该层。
        tgt_kv_cache_group = layer_to_kv_cache_group[target_layer_name]
        tgt_kv_cache_group.layer_names.append(layer_name)

        if runner_only_attn_layers is not None:
            # 标记该层是 runner 侧为了 metadata / forward context 管理补进去的，
            # 上层 KV cache manager 并不会为它单独分配 cache。
            runner_only_attn_layers.add(layer_name)


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """
    把已经分配好的 KV cache 同时绑定到 ModelRunner 和 forward context。

    本函数做两件事：
      1) 按层序号顺序填充 ModelRunner 的 `runner_kv_caches` 列表。
      2) 把 `forward_context` 中的每个 attention 层和它对应的 KV cache 关联起来。

    参数:
        kv_caches: 已分配好的 KV cache，key 是 attention 层名。
        forward_context: 全局 forward context，包含所有 attention 层。
        runner_kv_caches: ModelRunner 持有的 KV cache 列表。
    """
    # runner_kv_caches 必须还没绑定过，避免重复 append 造成层顺序错乱。
    assert len(runner_kv_caches) == 0

    # 先把 layer_name 按 layer_index 分组，再按层序号排序写入 runner_kv_caches。
    # 这样 runner 侧拿到的是稳定的“按模型层顺序排列”的 cache 列表。
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # 典型场景是 encoder-decoder 模型，例如 bart。
            # 同一个 decoder block 里 self-attention 和 cross-attention 的
            # layer_name 不同，但 layer_index 相同。

            # 待办：进一步分析 runner_kv_caches 的使用点，确定如何准确表达
            # 同一个 decoder block 里多个 attention 层的 cache。
            if (
                current_platform.is_cuda_alike()
                or current_platform.is_xpu()
                or current_platform.is_cpu()
            ):
                # 已知 GPU / CPU runner 不受这个场景影响。
                # 部分测试代码会读取 runner_kv_caches，但不会依赖这里被忽略的细节。
                pass
            else:
                raise NotImplementedError
        for layer_name in layer_names:
            # 每个实际 attention 层的 cache 都追加进 runner 持有的列表。
            runner_kv_caches.append(kv_caches[layer_name])

    # 再把 KV cache 绑定到 forward context 中对应的 Attention 模块。
    for layer_name, kv_cache in kv_caches.items():
        # 注意：这里用 list 是为了兼容 v0 pipeline-parallel virtual engine。
        forward_context[layer_name].kv_cache = [kv_cache]


def is_residual_scattered_for_sp(
    vllm_config: VllmConfig, num_input_tokens: int
) -> bool:
    """判断 residual tensor 是否已经按 sequence parallelism 被切分。

    当 sequence parallelism 和 tensor parallelism 同时启用时，residual tensor
    可能会按 token 维度切分到不同 tensor parallel rank 上。

    这里与 SequenceParallelismPass.is_applicable_for_range() 保持同一套判断逻辑：
    - full-graph 编译模式下（没有 splitting ops，或使用 inductor graph partition），
      总是应用 SP。
    - 否则，只有 num_input_tokens 落在 compile_sizes 中时才应用 SP。
    """
    if not vllm_config.compilation_config.pass_config.enable_sp:
        # 没开 sequence parallelism，自然不会切分 residual。
        return False

    tp = vllm_config.parallel_config.tensor_parallel_size

    if tp == 1:
        # tensor parallel size 为 1 时没有跨 rank 切分的对象。
        return False

    # 开启 SP 时，前面的输入准备阶段会把 num_input_tokens pad 到 tp 的倍数。
    assert num_input_tokens % tp == 0

    if (
        not vllm_config.compilation_config.splitting_ops
        or vllm_config.compilation_config.use_inductor_graph_partition
    ):
        # full-graph / inductor partition 路径下，SP pass 会稳定应用。
        return True
    compile_sizes = vllm_config.compilation_config.compile_sizes
    if compile_sizes is None:
        return False
    # 非 full-graph 场景下，只对预先编译过的形状应用 SP。
    return num_input_tokens in compile_sizes
