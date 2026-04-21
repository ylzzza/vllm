# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.ubatch_utils import UBatchSlices

logger = init_logger(__name__)

track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
# 记录“某个 batchsize 对应的多次 forward 耗时”，
# 用于周期性打印中位数统计，帮助观察不同批大小下的执行成本。
batchsize_forward_time: defaultdict = defaultdict(list)


@dataclass(frozen=True)
class BatchDescriptor:
    """描述 cudagraph dispatch 所需的批次形状特征。

    这个结构的目标不是完整复刻 batch 的所有状态，而是用尽量少的字段，
    唯一标识“padding 之后的批次形状”。这样 runtime 才能据此选择或复用
    对应的 CUDA graph。
    """

    num_tokens: int
    num_reqs: int | None = None
    """batch 中的请求数。

    对 PIECEWISE cudagraph 而言，图本身可以适配任意请求数，因此这里可以为
    `None`，表示“请求维度不是 dispatch key 的一部分”。
    """
    uniform: bool = False
    """若为 True，表示 batch 内所有请求本轮携带的 token 数完全一致。"""
    has_lora: bool = False
    """该 batch 是否启用了 LoRA 适配器。"""
    num_active_loras: int = 0
    """该 batch 中活跃的、互不相同的 LoRA 适配器数量。

    当开启 `cudagraph_specialize_lora_count` 时，会针对不同的
    `num_active_loras` 分别捕获 CUDA graph。这样像 `fused_moe_lora`
    这类 grid size 依赖 LoRA 数量的 kernel，才能被稳定地捕获和复用。
    """


def _compute_sp_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor, sequence_parallel_size: int
) -> list[int]:
    """把每个 DP rank 的 token 数换算成每个 SP 子 rank 的本地 token 数。

    做法是对每个 DP rank 的 token 数按 `sequence_parallel_size` 向上取整切分，
    然后用 `repeat_interleave` 展开成“每个 SP rank 各自处理多少 token”的列表。
    """
    sp_tokens = (
        num_tokens_across_dp_cpu + sequence_parallel_size - 1
    ) // sequence_parallel_size

    sp_tokens = sp_tokens.repeat_interleave(sequence_parallel_size)
    return sp_tokens.tolist()


def _compute_chunked_local_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor,
    sequence_parallel_size: int,
    max_num_tokens: int,
    chunk_idx: int,
) -> list[int]:
    """计算 chunked forward 时，每个 DP/SP rank 在当前 chunk 要处理多少 token。

    先得到未分 chunk 时每个 SP rank 的总 token 数，再取当前 `chunk_idx`
    对应的切片区间。即使某个 rank 已经没有剩余 token，也至少返回 1，
    以保证各 rank 仍能 lockstep 推进。
    """
    sp_tokens = _compute_sp_num_tokens(num_tokens_across_dp_cpu, sequence_parallel_size)
    sp_size = len(sp_tokens)

    local_size = [-1] * sp_size
    for i in range(sp_size):
        # 如果 MoE activation 经过了 sequence parallel 切分，
        # 每个 SP rank 只处理自己那份 token，再从中取出当前 chunk。
        local_size[i] = min(max_num_tokens, sp_tokens[i] - (max_num_tokens * chunk_idx))
        if local_size[i] <= 0:
            # 即使该 rank 已经没有真实工作量，也保留 1 个占位，
            # 保证并行 rank 在 chunked 执行中依旧同步前进。
            local_size[i] = 1
    return local_size


@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor
    num_tokens_across_dp_cpu: torch.Tensor

    # `local_sizes` 仅在 `chunked_sizes/sp_local_sizes` 上下文内有效，
    # 用来描述“当前这一小段执行里，每个 rank 实际应该消费多少 token”。
    local_sizes: list[int] | None = None

    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        num_tokens: int,
        num_tokens_across_dp_cpu: torch.Tensor,
    ) -> "DPMetadata":
        """根据当前 rank 与全局 DP token 分布，构造 `DPMetadata`。"""
        assert num_tokens_across_dp_cpu is not None
        assert parallel_config.data_parallel_size > 1
        assert parallel_config.is_moe_model is not False
        dp_rank = parallel_config.data_parallel_rank
        batchsize = num_tokens

        # `num_tokens_across_dp_cpu` 描述所有 DP rank 各自的 token 数。
        # 对当前 rank 而言，它记录的值必须与本地 `batchsize` 一致。
        assert num_tokens_across_dp_cpu[dp_rank] == batchsize, (
            f"{num_tokens_across_dp_cpu[dp_rank]} {batchsize}"
        )
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp_cpu)
        return DPMetadata(max_tokens_across_dp_cpu, num_tokens_across_dp_cpu)

    @contextmanager
    def chunked_sizes(
        self, sequence_parallel_size: int, max_chunk_size_per_rank: int, chunk_idx: int
    ):
        """在 chunked forward 期间，临时设置“当前 chunk 的每 rank token 数”。

        之所以要显式维护这份信息，是因为在 DP 场景下，各 rank 的 token 数
        可能不完全一致；进一步叠加 SP 与 chunking 之后，某些 rank 甚至会更早
        跑完自己的真实输入。这里通过统一计算每个 rank 在当前 chunk 该处理的
        token 数，并把结果暂存在 `self.local_sizes`，保证并行执行仍能 lockstep。

        计算逻辑：
        1. 先按 SP 维度把每个 DP rank 的 token 数切开；
        2. 再取第 `chunk_idx` 段，每段最多 `max_chunk_size_per_rank` 个 token；
        3. 若某个 rank 已经没有剩余 token，则保留 1 个占位 token。

        `self.local_sizes` 只在 `with` 代码块内部有效，退出后会恢复为 `None`。

        参数：
            sequence_parallel_size:
                当 attention 走 TP、MoE 层走 EP 时，中间会用 SP 避免冗余计算，
                这里需要这个值来计算切分后的本地 token 数。
            max_chunk_size_per_rank:
                当前 chunk 中，单个 rank 最多允许处理多少 token。
            chunk_idx:
                当前是第几个 chunk，从 0 开始计数。
        """
        self.local_sizes = _compute_chunked_local_num_tokens(
            self.num_tokens_across_dp_cpu,
            sequence_parallel_size,
            max_chunk_size_per_rank,
            chunk_idx,
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    @contextmanager
    def sp_local_sizes(self, sequence_parallel_size: int):
        """临时设置未做 chunking 时的每个 SP rank 本地 token 数。

        它和 `chunked_sizes()` 的语义一致，只是少了 chunk 切分这一步，
        直接把每个 DP rank 的 token 数按 SP 均摊后写入 `self.local_sizes`。
        """
        self.local_sizes = _compute_sp_num_tokens(
            self.num_tokens_across_dp_cpu, sequence_parallel_size
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None:
        assert self.local_sizes is not None
        return self.local_sizes

    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor:
        """返回跨 SP rank 的 token 前缀和。

        先把每个 DP rank 的 token 数按 `sp_size` 向上取整切到各 SP rank，
        再做 `cumsum`。结果常用于描述 MoE 输入在 DP 与 TP/SP 共同分布时的
        全局分段边界。

        当 `sp_size == 1` 时，它就退化成单纯的“跨 DP rank token 前缀和”。
        """
        num_tokens_across_sp_cpu = (
            self.num_tokens_across_dp_cpu - 1 + sp_size
        ) // sp_size
        num_tokens_across_sp_cpu = num_tokens_across_sp_cpu.repeat_interleave(sp_size)
        return torch.cumsum(num_tokens_across_sp_cpu, dim=0)


@dataclass
class ForwardContext:
    # 从 `vllm_config.compilation_config.static_forward_context` 复制而来，
    # 表示这轮 forward 中哪些层不应走 compile 包装。
    no_compile_layers: dict[str, Any]
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
    """当前 forward 对 attention 层所需运行时信息的统一入口。

    `attn_metadata`:
        v1 普通路径下为 `Dict[str, AttentionMetadata]`，把 layer name 映射到
        该层要使用的 attention metadata。

        DBO/ubatching 路径下为 `List[Dict[str, AttentionMetadata]]`，
        列表中的每一项对应一个 microbatch/ubatch。

    `slot_mapping`:
        结构与 `attn_metadata` 对齐，保存每层或每个 ubatch 对应的 slot mapping。

    这些字段都会在每次 forward 前动态设置。
    """
    # TODO: 等所有 virtual_engine 共享同一份 kv cache 后删除。
    virtual_engine: int  # 每次 forward 动态设置，用于区分虚拟执行通道。
    # 每次 forward 动态设置；仅在 DP+MoE 等场景下才会有值。
    dp_metadata: DPMetadata | None = None
    # 运行时决定当前这一轮 forward 使用哪种 cudagraph 模式：
    # FULL、PIECEWISE 或 NONE。默认 NONE，表示不使用 cudagraph。
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_descriptor: BatchDescriptor | None = None

    # 开启 DBO/ubatching 时，描述每个 ubatch 在 request/token 维度上的切片。
    ubatch_slices: UBatchSlices | None = None

    # 为 True 时跳过 compiled model 包装，直接调用原始 forward。
    skip_compiled: bool = False

    # 为了降低 torch.compile 冷启动成本，需要避免把字符串常量直接烘焙进图。
    # 当前 `vllm.moe_forward` / `vllm.moe_forward_shared` 这些自定义算子
    # 仍然需要 layer name 字符串，因此这里采用折中方案：
    #
    # 1. 在 ForwardContext 中预存本轮会用到的 `all_moe_layers`
    # 2. 维护一个递增游标 `moe_layer_index`
    # 3. 自定义算子执行时，从列表里取“下一个 layer name”
    #
    # 这依赖一个前提：这些自定义算子在运行时的执行顺序稳定，且不会被
    # torch.compile 重新排序。
    #
    # TODO(https://github.com/vllm-project/vllm/issues/31985):
    # 更彻底的方案还在推进中，例如拆开 moe custom operator，或者把字符串
    # 当作 graph 的 symbolic input；但 PyTorch 侧目前还没完全就绪。
    #
    # 如果这里是 `None`（例如某些测试场景），字符串就会被直接固化进图；
    # 否则 MoE custom op 会从这个列表中按顺序取值。
    all_moe_layers: list[str] | None = None
    # 指向 `all_moe_layers` 中“下一个待消费 layer name”的游标。
    moe_layer_index: int = 0

    # 平台相关扩展字段，供不同 device/backend 注入额外上下文。
    additional_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # `ForwardContext` 会被大量运行时路径依赖，进入系统前先校验模式合法。
        assert self.cudagraph_runtime_mode.is_valid_runtime_mode(), (
            f"Invalid cudagraph runtime mode: {self.cudagraph_runtime_mode}"
        )


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    """返回当前线程内正在生效的 forward context。"""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


def is_forward_context_available() -> bool:
    """当前是否已经设置了 forward context。"""
    return _forward_context is not None


def create_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    dp_metadata: DPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    additional_kwargs: dict[str, Any] | None = None,
    skip_compiled: bool = False,
):
    """根据本轮 forward 的运行时输入，构造 `ForwardContext` 对象。"""
    if vllm_config.compilation_config.fast_moe_cold_start:
        # cold start 优化开启时，预先把静态的 MoE layer name 列表放进 context，
        # 避免自定义算子把字符串常量直接编进 graph。
        all_moe_layers = vllm_config.compilation_config.static_all_moe_layers
    else:
        all_moe_layers = None

    return ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        all_moe_layers=all_moe_layers,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping or {},
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
        skip_compiled=skip_compiled,
        additional_kwargs=additional_kwargs or {},
    )


@contextmanager
def override_forward_context(forward_context: ForwardContext | None):
    """临时覆盖当前生效的 forward context。

    常用于某次特定 forward 想显式替换上下文时；退出 `with` 后会恢复旧值。
    """
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    skip_compiled: bool = False,
):
    """为一次模型 forward 安装运行时上下文。

    这个上下文管理器是模型前向执行的统一入口之一。它负责：
    1. 挂载本轮的 attention metadata / slot mapping；
    2. 在 DP+MoE 场景下补齐 `DPMetadata`；
    3. 根据需要生成 `BatchDescriptor`；
    4. 让平台层注入额外的 forward 上下文字段；
    5. 在 forward 结束后记录 batchsize 与耗时统计。
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()

    dp_metadata: DPMetadata | None = None
    if (
        vllm_config.parallel_config.data_parallel_size > 1
        and vllm_config.parallel_config.is_moe_model is not False
        and (attn_metadata is not None or num_tokens is not None)
    ):
        # 如果调用方没有提前给出 `num_tokens_across_dp`，就在这里现场协调一次。
        # 这里仅为了拿到各 DP rank 的 token 分布，因此禁用 DP padding 和
        # microbatching，避免引入额外形状变化。
        if num_tokens_across_dp is None:
            assert ubatch_slices is None
            assert num_tokens is not None
            _, num_tokens_across_dp, _ = coordinate_batch_across_dp(
                num_tokens_unpadded=num_tokens,
                parallel_config=vllm_config.parallel_config,
                allow_microbatching=False,
            )
            assert num_tokens_across_dp is not None
        dp_metadata = DPMetadata.make(
            vllm_config.parallel_config, num_tokens or 0, num_tokens_across_dp
        )

    # 便捷逻辑：如果启用了 cudagraph 且提供了 `num_tokens`，
    # 但调用方还没给 `batch_descriptor`，这里可以先构一个最基础的版本。
    # 即使后续 wrapper 发现它不匹配，也会正常 fallback，不会影响正确性。
    if cudagraph_runtime_mode != CUDAGraphMode.NONE and num_tokens is not None:
        batch_descriptor = batch_descriptor or BatchDescriptor(num_tokens=num_tokens)

    # 不同平台可能需要为 forward 再注入一些额外上下文，例如设备相关开关、
    # backend 专用运行时字段等。
    additional_kwargs = current_platform.set_additional_forward_context(
        attn_metadata=attn_metadata,
        vllm_config=vllm_config,
        virtual_engine=virtual_engine,
        dp_metadata=dp_metadata,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
    )

    forward_context = create_forward_context(
        attn_metadata,
        vllm_config,
        virtual_engine,
        dp_metadata,
        cudagraph_runtime_mode,
        batch_descriptor,
        ubatch_slices,
        slot_mapping,
        additional_kwargs,
        skip_compiled,
    )

    try:
        with override_forward_context(forward_context):
            yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            batchsize = num_tokens
            # 这里统计的是“完整 forward 的真实结束时间”，因此如果平台提供了
            # synchronize，就先把异步 CUDA 工作对齐后再读时钟。
            # 当前调度是同步推进的，所以这里插入同步点不会影响下一批调度。
            synchronize = current_platform.synchronize
            if synchronize is not None:
                synchronize()
            now = time.perf_counter()
            # 统一把耗时记录成毫秒。
            batchsize_forward_time[batchsize].append((now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # 单样本通常不稳定，可能只是 cudagraph 捕获或 profiling run。
                        continue
                    median = torch.quantile(torch.tensor(times), q=0.5).item()
                    median = round(median, 2)
                    # 输出项格式为：
                    # `(batchsize, 观测次数, 中位耗时ms)`。
                    forward_stats.append((bs, len(times), median))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(
                        (
                            "Batchsize forward time stats "
                            "(batchsize, count, median_time(ms)): %s"
                        ),
                        forward_stats,
                    )
