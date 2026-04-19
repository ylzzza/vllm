# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import Field, field_validator, model_validator
from torch.distributed import ProcessGroup, ReduceOp
from typing_extensions import Self

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_ports_list
from vllm.utils.torch_utils import cuda_device_count_stateless

if TYPE_CHECKING:
    from ray.runtime_env import RuntimeEnv
    from ray.util.placement_group import PlacementGroup

    from vllm.v1.executor import Executor
else:
    RuntimeEnv = Any
    PlacementGroup = Any
    Executor = Any

logger = init_logger(__name__)

ExpertPlacementStrategy = Literal["linear", "round_robin"]
DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]
DataParallelBackend = Literal["ray", "mp"]
EPLBPolicyOption = Literal["default"]
All2AllBackend = Literal[
    "naive",
    "pplx",
    "deepep_high_throughput",
    "deepep_low_latency",
    "mori",
    "allgather_reducescatter",
    "flashinfer_all2allv",
]


@config
class EPLBConfig:
    """Expert Parallel Load Balancing（EPLB，专家并行负载均衡）的配置。"""

    window_size: int = 1000
    """记录专家负载指标时使用的滑动窗口大小。"""
    step_interval: int = 3000
    """
    专家并行中重新排列专家的 step 间隔。

    如果该值大于 EPLB window_size，那么重新排列专家时只会使用最近
    `window_size` 个 step 的负载指标。
    """

    num_redundant_experts: int = Field(default=0, ge=0)
    """专家并行中使用的冗余专家数量。"""

    log_balancedness: bool = False
    """
    是否在专家并行的每个 step 记录负载均衡度。
    默认关闭，因为它会引入额外通信开销。
    """
    log_balancedness_interval: int = 1
    """
    记录负载均衡度的 step 间隔。
    """
    use_async: bool = False
    """
    是否使用非阻塞 EPLB。
    """

    policy: EPLBPolicyOption = "default"
    """EPLB 使用的策略类型。"""

    @model_validator(mode="after")
    def _validate_eplb_config(self) -> Self:
        # 异步 EPLB 当前只实现了默认策略，其它策略组合直接拒绝。
        if self.use_async and self.policy != "default":
            raise ValueError("Async EPLB is only supported with the default policy.")
        # 开启日志时，间隔必须为正数，否则运行时无法按 step 取模。
        if self.log_balancedness and self.log_balancedness_interval <= 0:
            raise ValueError("log_balancedness_interval must be greater than 0.")
        return self


@config
class ParallelConfig:
    """分布式执行相关配置。"""

    pipeline_parallel_size: int = 1
    """pipeline parallel（PP）的并行大小。"""
    tensor_parallel_size: int = 1
    """tensor parallel（TP）的并行大小。"""
    prefill_context_parallel_size: int = 1
    """prefill context parallel（PCP）的并行大小。"""
    data_parallel_size: int = 1
    """data parallel（DP）的并行大小。MoE 层会按 TP size 和 DP size 的乘积切分。"""
    data_parallel_size_local: int = 1
    """本节点上的本地 DP size。"""
    data_parallel_rank: int = 0
    """当前进程所在 DP group 的 rank。"""
    data_parallel_rank_local: int | None = None
    """当前进程在本地 DP group 中的 rank；只在 SPMD 模式下设置。"""
    data_parallel_master_ip: str = "127.0.0.1"
    """DP master 的 IP 地址。"""
    data_parallel_rpc_port: int = 29550
    """DP 消息通信使用的端口。"""
    data_parallel_master_port: int = 29500
    """DP master 使用的端口。"""
    data_parallel_backend: DataParallelBackend = "mp"
    """DP 使用的 backend，可选 "mp" 或 "ray"。"""
    data_parallel_external_lb: bool = False
    """是否使用 external DP 负载均衡模式。只适用于在线服务且 data_parallel_size > 0。
    该模式适合 Kubernetes 中“每个 rank 一个 pod”的 wide-EP 部署。
    当 vllm serve 显式提供 --data-parallel-rank 时，会隐式启用该模式。"""
    data_parallel_hybrid_lb: bool = False
    """是否使用 hybrid DP 负载均衡模式。只适用于在线服务且 data_parallel_size > 0。
    该模式允许每个节点运行一组 AsyncLLM 和 API server：vLLM 负责在本地 DP rank
    之间做负载均衡，外部 LB 负责在不同 vLLM 节点/副本之间做负载均衡。
    需要和 --data-parallel-start-rank 一起显式设置。"""
    is_moe_model: bool | None = None
    """部署的模型是否为 MoE；未知时为 None。"""
    enable_expert_parallel: bool = False
    """MoE 层是否使用 expert parallel，而不是普通 tensor parallel。"""
    enable_eplb: bool = False
    """是否为 MoE 层启用专家并行负载均衡。"""
    eplb_config: EPLBConfig = Field(default_factory=EPLBConfig)
    """专家并行负载均衡配置。"""
    expert_placement_strategy: ExpertPlacementStrategy = "linear"
    """MoE 层的专家放置策略：\n
    - "linear": 连续放置专家。例如 4 个专家、2 个 rank 时，rank 0 持有
      [0, 1]，rank 1 持有 [2, 3]。\n
    - "round_robin": 轮询放置专家。例如 4 个专家、2 个 rank 时，rank 0 持有
      [0, 2]，rank 1 持有 [1, 3]。在没有冗余专家的 grouped expert 模型中，
      该策略有助于改善负载均衡。"""
    all2all_backend: All2AllBackend = "allgather_reducescatter"
    """MoE expert parallel 通信使用的 All2All backend。可选项：

    - "naive": 基于 broadcast 的朴素 all2all 实现\n
    - "allgather_reducescatter": 基于 allgather + reducescatter 的 all2all\n
    - "deepep_high_throughput": 使用 DeepEP 高吞吐 kernel\n
    - "deepep_low_latency": 使用 DeepEP 低延迟 kernel\n
    - "mori": 使用 mori kernel\n
    - "flashinfer_all2allv": 为 mnnvl 使用 flashinfer alltoallv kernel"""

    max_parallel_loading_workers: int | None = None
    """模型分批顺序加载时允许的最大并行加载 worker 数。
    用于在 TP + 大模型场景下降低 CPU RAM OOM 风险。"""

    disable_custom_all_reduce: bool = False
    """是否禁用自定义 all-reduce kernel，并回退到 NCCL。"""

    enable_elastic_ep: bool = False
    """是否启用 elastic expert parallelism，并为 DP/EP 使用 stateless NCCL groups。"""

    enable_dbo: bool = False
    """是否为 model executor 启用 dual batch overlap（DBO）。"""
    ubatch_size: int = 0
    """ubatch 的数量/大小配置。"""

    dbo_decode_token_threshold: int = 32
    """只包含 decode 的 batch 启用 DBO 的 token 数阈值。
    如果请求中的 token 数大于该阈值，会启用 microbatching；
    否则直接按单个 batch 处理。"""
    dbo_prefill_token_threshold: int = 512  # 待调优。
    """包含一个或多个 prefill 的 batch 启用 DBO 的 token 数阈值。
    如果请求中的 token 数大于该阈值，会启用 microbatching；
    否则直接按单个 batch 处理。"""

    disable_nccl_for_dp_synchronization: bool | None = Field(default=None)
    """强制 vllm/v1/worker/dp_utils.py 中的 DP 同步逻辑使用 Gloo 而不是 NCCL
    执行 all-reduce。

    启用 async scheduling 时默认 True，否则默认 False。
    """

    ray_workers_use_nsight: bool = False
    """是否使用 nsight 分析 Ray workers。参考 Ray 的 nsight profiling 文档。"""

    ray_runtime_env: RuntimeEnv | None = None
    """传给分布式 worker 的 Ray runtime environment。"""

    placement_group: PlacementGroup | None = None
    """Ray 分布式模型 worker 使用的 placement group。"""

    distributed_executor_backend: (
        str | DistributedExecutorBackend | type[Executor] | None
    ) = None
    """分布式模型 worker 使用的 backend，可以是 "ray" 或 "mp"（multiprocessing）。
    如果 pipeline_parallel_size 与 tensor_parallel_size 的乘积不超过当前可用 GPU 数，
    默认会使用 "mp"，让处理留在单机内。否则会报错。
    使用 "mp" 多机时还需要设置 nnodes；使用 "ray" 时需要手动把
    distributed_executor_backend 设置为 "ray"。

    注意：TPU 分布式推理只支持 Ray。"""

    worker_cls: str = "auto"
    """使用的 worker 类完整名称。若为 "auto"，会根据平台自动选择 worker 类。"""
    sd_worker_cls: str = "auto"
    """speculative decoding 使用的 worker 类完整名称。
    若为 "auto"，会根据平台自动选择 worker 类。"""
    worker_extension_cls: str = ""
    """使用的 worker extension 类完整名称。
    worker 类会动态继承该 extension 类，用于向 worker 注入新的属性和方法，
    供 collective_rpc 调用使用。"""
    master_addr: str = "127.0.0.1"
    """distributed_executor_backend 为 mp 时，多机分布式推理使用的 master 地址。"""
    master_port: int = 29501
    """distributed_executor_backend 为 mp 时，多机分布式推理使用的 master 端口。"""
    node_rank: int = 0
    """distributed_executor_backend 为 mp 时，多机分布式推理中的节点 rank。"""
    nnodes: int = 1
    """distributed_executor_backend 为 mp 时，多机分布式推理的节点数。"""

    world_size: int = Field(init=False)
    """world_size 等于 TP x PP x PCP，会影响需要创建的 worker 数量。"""

    rank: int = 0
    """分布式环境中的全局 rank。"""

    _data_parallel_master_port_list: list[int] = Field(default_factory=list)
    """为 DP 消息通信自动查询到的可用端口列表。
    该字段是内部私有配置，不应由用户直接设置。
    """

    _stateless_dp_group_port_list: list[list[int]] = Field(default_factory=list)
    """enable_elastic_ep 为 True 时，stateless DP groups 使用的可用端口列表。
    该字段是内部私有配置，不应由用户直接设置。
    这是一个 list[list[int]]，其中每个内部 list 包含 3 个端口，分别用于
    StatelessGroupCoordinator 中的 stateless CPU/device/TCPStore groups。
    内部 list 的数量等于 DP group 数量，即
    len(self._stateless_dp_group_port_list) == world_size_across_dp // dp_size，
    且每个内部 list 的长度都是 3。
    """

    _stateless_ep_group_port_list: list[list[int]] = Field(default_factory=list)
    """enable_elastic_ep 为 True 时，stateless EP groups 使用的可用端口列表。
    该字段是内部私有配置，不应由用户直接设置。
    len(self._stateless_ep_group_port_list) == world_size_across_dp // ep_size。
    """

    _stateless_eplb_group_port_list: list[list[int]] = Field(default_factory=list)
    """enable_elastic_ep 为 True 时，stateless EPLB groups 使用的可用端口列表。
    它与 EP 使用相同拓扑，但使用独立 NCCL communicator 以避免死锁。
    """

    _stateless_world_group_port_list: list[list[int]] = Field(default_factory=list)
    """enable_elastic_ep 为 True 时，stateless world group 使用的可用端口列表。
    该字段是内部私有配置，不应由用户直接设置。
    len(self._stateless_world_group_port_list) == 1。
    """

    decode_context_parallel_size: int = 1
    """decode context parallel（DCP）的并行大小。
    DCP 不改变 world size，只复用 TP group 内的 GPU，因此 tp_size 必须能被
    dcp_size 整除。"""

    dcp_kv_cache_interleave_size: int = 1
    """
    使用 DCP 时 KV cache 存储的交错粒度。
    dcp_kv_cache_interleave_size 已被 cp_kv_cache_interleave_size 替代，
    PCP 完全支持后该字段会废弃。

    """
    cp_kv_cache_interleave_size: int = 1
    """使用 DCP 或 PCP 时 KV cache 存储的交错粒度。
    对于 `total_cp_rank = pcp_rank * dcp_world_size + dcp_rank`，
    以及 `total_cp_world_size = pcp_world_size * dcp_world_size`：
    会先在 total_cp_rank i 上存 interleave_size 个 token，
    再把接下来的 interleave_size 个 token 存到 total_cp_rank i+1。
    interleave_size=1 表示 token 级对齐，token `i` 存在
    total_cp_rank `i % total_cp_world_size` 上。
    interleave_size=block_size 表示 block 级对齐：token 会先填满前面的 rank，
    只有 (rank i, block j) 填满后，才会写入 (rank i+1, block j)。
    block_size 必须大于等于 cp_kv_cache_interleave_size，
    且必须能被 cp_kv_cache_interleave_size 整除。
    """

    data_parallel_index: int = Field(init=False)
    """等于 data_parallel_rank，但不用于 torch process group，
    且 dense 模型不会覆盖它。"""

    _api_process_count: int = Field(default=1, gt=0)
    """
    已初始化的 API 进程数量。

    注意：
        这是内部配置，只对 API server scale-out 有效，也只应由该路径设置。
    """

    _api_process_rank: int = Field(default=0, ge=-1)
    """
    当前 API 进程的 rank；在 API server scale-out 下，engine core 进程使用 `-1`。

    注意：
        这是内部配置，只对 API server scale-out 有效，也只应由该路径设置。
    """

    @field_validator("disable_nccl_for_dp_synchronization", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """延迟初始化时，如果值为 `None`，就跳过该字段校验。"""
        return None if value is None else handler(value)

    @model_validator(mode="after")
    def _validate_parallel_config(self) -> Self:
        # API 进程 rank 只能是 engine core 的 -1，或者落在 API 进程数量范围内。
        if self._api_process_rank >= self._api_process_count:
            raise ValueError(
                "Invalid value of `_api_process_rank`. "
                f"Expected to be `-1` or `[0, {self._api_process_count})`, "
                f"but found: {self._api_process_rank}"
            )

        if self.all2all_backend == "pplx":
            # pplx backend 已被移除，保留兼容路径并自动回退到默认可用实现。
            logger.warning(
                "The 'pplx' all2all backend has been removed. "
                "Falling back to 'allgather_reducescatter'."
            )
            self.all2all_backend = "allgather_reducescatter"

        # 本地 DP 数不能超过全局 DP 数，否则 rank 拓扑无法成立。
        if self.data_parallel_size_local > self.data_parallel_size:
            raise ValueError(
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must be <= data_parallel_size ({self.data_parallel_size})"
            )

        # external LB 只有在实际存在多个 DP rank 时才有意义。
        if self.data_parallel_size <= 1 and self.data_parallel_external_lb:
            raise ValueError(
                "data_parallel_external_lb can only be set when data_parallel_size > 1"
            )

        if self.enable_eplb:
            # EPLB 目前依赖 CUDA/ROCm 能力，其它平台直接拒绝。
            if not current_platform.is_cuda_alike():
                raise ValueError(
                    "Expert parallelism load balancing is only supported on "
                    "CUDA devices or ROCm devices now."
                )
            # EPLB 是 expert parallel 的负载均衡层，不能单独开启。
            if not self.enable_expert_parallel:
                raise ValueError("enable_expert_parallel must be True to use EPLB.")
            # TP*DP 至少要大于 1，否则没有跨 rank 专家放置/均衡的意义。
            if self.tensor_parallel_size * self.data_parallel_size <= 1:
                raise ValueError(
                    "EPLB requires tensor_parallel_size or data_parallel_size "
                    f"to be greater than 1, but got "
                    f"TP={self.tensor_parallel_size},DP={self.data_parallel_size}."
                )
        else:
            # 配置冗余专家但没开 EPLB 会造成语义不一致，提前报错。
            if self.eplb_config.num_redundant_experts != 0:
                raise ValueError(
                    "num_redundant_experts is set to "
                    f"{self.eplb_config.num_redundant_experts} but EPLB is not "
                    "enabled. Either enable EPLB or unset "
                    "num_redundant_experts."
                )

        # 当前 DCP 实现不会改变 world size，而是复用 TP group 内的 GPU，
        # 并把一个 TP group 拆成 tp_size // dcp_size 个 DCP group。
        # 因此 tp_size 必须能被 dcp_size 整除。
        if self.tensor_parallel_size % self.decode_context_parallel_size != 0:
            raise ValueError(
                f"tp_size={self.tensor_parallel_size} must be divisible by"
                f"dcp_size={self.decode_context_parallel_size}."
            )

        return self

    @property
    def world_size_across_dp(self) -> int:
        """包含 DP 后的总 world size，即 TP x PP x PCP x DP。"""
        return self.world_size * self.data_parallel_size

    @property
    def use_ubatching(self) -> bool:
        # DBO 会隐式使用 ubatching；显式设置 ubatch_size > 1 也会启用。
        return self.enable_dbo or self.ubatch_size > 1

    @property
    def num_ubatches(self) -> int:
        # DBO 当前固定拆成两个 ubatch；否则使用用户配置的 ubatch_size。
        return 2 if self.enable_dbo else self.ubatch_size

    @property
    def local_engines_only(self) -> bool:
        """
        纯 internal LB 场景下，client 会管理本地和远端的 EngineCores。
        hybrid/external LB 场景下，client 只管理本地 EngineCores。
        """
        return self.data_parallel_external_lb or self.data_parallel_hybrid_lb

    def get_next_dp_init_port(self) -> int:
        """
        与 DP 相关的 process group 可能需要在多个进程中初始化，例如 worker
        和 engine 可能位于不同进程。为避免端口冲突，每次初始化新的 DP 相关
        process group 时，都从预先准备的端口列表中取一个新端口。
        """
        if self._data_parallel_master_port_list:
            # 优先使用预分配端口，避免多个进程临时抢同一个空闲端口。
            answer = self._data_parallel_master_port_list.pop()
        else:
            # 没有预分配端口时，按当前 master port 递增兜底。
            answer = self.data_parallel_master_port
            self.data_parallel_master_port += 1

        return answer

    def allocate_elastic_ep_ports(self) -> None:
        """为 elastic EP 分配全部端口，包括 stateless groups 和 DP master。

        必须在 ray.init() 之后调用。这样 Ray idle worker pool 已经占用的端口
        不会再被 get_open_ports_list() 返回，降低端口冲突概率。
        """
        if not self.enable_elastic_ep:
            return
        if self._stateless_world_group_port_list:
            # 已经分配过则直接返回，避免重复消耗端口列表。
            return

        # stateless group 每组需要 3 个端口，分别服务 CPU/device/TCPStore group。
        num_world_groups = 1
        dp_size = self.data_parallel_size
        ep_size = self.data_parallel_size * self.world_size_across_dp
        num_dp_groups = max(1, self.world_size_across_dp // dp_size)
        num_ep_groups = max(1, self.world_size_across_dp // ep_size)
        num_eplb_groups = num_ep_groups
        total_stateless_ports = (
            num_world_groups + num_dp_groups + num_ep_groups + num_eplb_groups
        ) * 3
        # 额外预留几个 DP master 端口，供后续多次初始化 DP 相关 group 使用。
        num_dp_master_ports = 5

        all_ports = get_open_ports_list(total_stateless_ports + num_dp_master_ports)

        # 端口列表尾部预留给 DP master；当前 master_port 先取一个。
        self._data_parallel_master_port_list = all_ports[-num_dp_master_ports:]
        self.data_parallel_master_port = self._data_parallel_master_port_list.pop()
        all_ports = all_ports[:-num_dp_master_ports]

        # 剩余端口按每 3 个一组切给 world / DP / EP / EPLB stateless groups。
        self._stateless_world_group_port_list = [
            all_ports[i : i + 3] for i in range(0, num_world_groups * 3, 3)
        ]
        start_idx = num_world_groups * 3
        self._stateless_dp_group_port_list = [
            all_ports[i : i + 3]
            for i in range(start_idx, start_idx + num_dp_groups * 3, 3)
        ]
        start_idx += num_dp_groups * 3
        self._stateless_ep_group_port_list = [
            all_ports[i : i + 3]
            for i in range(start_idx, start_idx + num_ep_groups * 3, 3)
        ]
        start_idx += num_ep_groups * 3
        self._stateless_eplb_group_port_list = [
            all_ports[i : i + 3]
            for i in range(start_idx, start_idx + num_eplb_groups * 3, 3)
        ]

    def get_next_stateless_world_group_port(self) -> list[int]:
        return self._stateless_world_group_port_list.pop()

    def get_next_stateless_dp_group_port(self) -> list[int]:
        return self._stateless_dp_group_port_list.pop()

    def get_next_stateless_ep_group_port(self) -> list[int]:
        return self._stateless_ep_group_port_list.pop()

    def get_next_stateless_eplb_group_port(self) -> list[int]:
        return self._stateless_eplb_group_port_list.pop()

    def stateless_init_dp_group(self, return_store: bool = False) -> ProcessGroup:
        # 高并发场景下，多个进程可能在调用 get_open_port() 时因为竞争条件拿到同一个
        # “当前空闲”的端口。第一个进程 bind 成功后，其它进程会因 EADDRINUSE 失败。
        # 因此这里遇到该错误时会换一个新端口重试几次，提高初始化鲁棒性。
        from torch.distributed import DistNetworkError

        from vllm.distributed.utils import (
            stateless_init_torch_distributed_process_group,
        )

        max_retries = 5
        last_exc: Exception | None = None
        for _ in range(max_retries):
            try:
                # engine 进程可能没有 CUDA 设备，因此这里使用 gloo。
                return stateless_init_torch_distributed_process_group(
                    self.data_parallel_master_ip,
                    self.get_next_dp_init_port(),
                    self.data_parallel_rank,
                    self.data_parallel_size,
                    backend="gloo",
                    return_store=return_store,
                )
            except DistNetworkError as e:
                # 只在根因是 EADDRINUSE 时重试；其它网络错误直接抛出。
                if "EADDRINUSE" in str(e):
                    logger.warning("Address already in use. Retrying with a new port.")
                    last_exc = e
                    continue  # 换一个新端口后重试。
                raise e

        # 走到这里说明所有重试都失败了，抛出最后一次异常。
        assert last_exc is not None
        raise last_exc

    # attention 末尾 o_proj 阶段的 all_reduce 会让输入在 tensor parallel group
    # 的每个 rank 上都保留一份副本。
    # 如果同时使用 expert parallel 和 DeepEP All2All，这些重复 token 会导致
    # 无意义的重复计算和通信。
    #
    # 因此在该场景下，需要确保进入 experts 的输入是 sequence parallel 的，
    # 以避免额外工作。
    #
    @property
    def use_sequence_parallel_moe(self) -> bool:
        return (
            self.all2all_backend
            in (
                "allgather_reducescatter",
                "naive",
                "deepep_high_throughput",
                "deepep_low_latency",
                "mori",
            )
            and self.enable_expert_parallel
            and self.tensor_parallel_size > 1
            and self.data_parallel_size > 1
        )

    @property
    def node_rank_within_dp(self) -> int:
        # 多机 DP 场景下，计算当前节点在所属 DP 分组内部的节点 rank。
        return self.node_rank % self.nnodes_within_dp

    @property
    def nnodes_within_dp(self) -> int:
        if self.nnodes == 1:
            return 1
        # 每个 DP 副本横跨的节点数 = 总节点数 / DP 节点组数量。
        data_parallel_node_size = (
            self.data_parallel_size // self.data_parallel_size_local
        )
        return self.nnodes // data_parallel_node_size

    @property
    def local_world_size(self) -> int:
        # 当前 DP 分组内每个节点实际需要启动的 worker 数。
        return self.world_size // self.nnodes_within_dp

    @staticmethod
    def has_unfinished_dp(dp_group: ProcessGroup, has_unfinished: bool) -> bool:
        tensor = torch.tensor([has_unfinished], dtype=torch.int32, device="cpu")
        # 示例：
        # dp rank 0: has_unfinished_seqs=True
        # dp rank 1: has_unfinished_seqs=False
        # 聚合结果: has_unfinished_seqs=True
        # 因此这里等价于 OR 操作，用整数 MAX 实现。
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        return aggregated_has_unfinished

    @staticmethod
    def sync_kv_cache_memory_size(dp_group: ProcessGroup, kv_cache_memory: int) -> int:
        # -1 表示当前 rank 没有有效值。这里把它替换成 int64 最大值，
        # 这样后面的 MIN all-reduce 会自动忽略它。
        if kv_cache_memory == -1:
            kv_cache_memory = torch.iinfo(torch.int64).max
        tensor = torch.tensor([kv_cache_memory], dtype=torch.int64, device="cpu")
        # stateless DP group 不能使用 broadcast，因为 broadcast 依赖全局 rank。
        # 用 MIN all-reduce 可以让所有 rank 拿到一致的最小可用 KV cache 显存值。
        torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=dp_group)
        return tensor.item()

    def compute_hash(self):
        """
        生成一个 hash，用于唯一标识会影响“从 input ids/embeddings 到最终
        hidden states”这段计算图结构的配置。

        不包含 input ids/embeddings 之前的逻辑，也不包含最终 hidden states 之后的逻辑。

        该 hash 也用于 DP worker 的配置一致性校验，避免不同 worker 因 collective
        communication pattern 不一致而 hang 住。
        """
        ignored_factors = {
            # 以下字段属于派生拓扑、运行时网络或启动细节，不影响模型计算图结构。
            "data_parallel_rank",
            "data_parallel_rank_local",
            "data_parallel_size_local",
            "data_parallel_index",
            "data_parallel_backend",
            "data_parallel_external_lb",
            "data_parallel_hybrid_lb",
            "data_parallel_master_ip",
            "data_parallel_master_port",
            "_data_parallel_master_port_list",
            "data_parallel_rpc_port",
            "rank",
            "master_addr",
            "master_port",
            "node_rank",
            "nnodes",
            "max_parallel_loading_workers",
            "disable_custom_all_reduce",
            "ray_workers_use_nsight",
            "ray_runtime_env",
            "placement_group",
            "distributed_executor_backend",
            "worker_cls",
            "sd_worker_cls",
            "worker_extension_cls",
            "_api_process_count",
            "_api_process_rank",
        }

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    def __post_init__(self) -> None:
        # 根据 PP、TP、PCP 计算基础 world_size。DP 默认不计入这里，
        # 除非使用 external_launcher。
        self.world_size = (
            self.pipeline_parallel_size
            * self.tensor_parallel_size
            * self.prefill_context_parallel_size
        )

        if self.distributed_executor_backend == "external_launcher":
            # external launcher 已经在外部创建完整进程拓扑，因此这里把 DP 也计入。
            logger.info("Using external launcher for distributed inference.")
            self.world_size *= self.data_parallel_size

        if self.enable_elastic_ep:
            # elastic EP 依赖 EPLB 的状态管理能力。
            if not self.enable_eplb:
                raise ValueError("Elastic EP is only supported with enable_eplb=True.")
            # 当前 elastic EP 不支持 pipeline parallel，否则扩缩容时拓扑过于复杂。
            if self.pipeline_parallel_size > 1:
                raise ValueError(
                    "Elastic EP is not supported with pipeline parallelism "
                    f"(pipeline_parallel_size={self.pipeline_parallel_size})."
                )
            # elastic EP 需要单个 API server/core client 统一协调扩缩容，
            # 因此不能和 external/hybrid LB 同时使用。
            if self.data_parallel_external_lb or self.data_parallel_hybrid_lb:
                raise NotImplementedError(
                    "Elastic EP is not compatible with data_parallel_external_lb "
                    "or data_parallel_hybrid_lb. Elastic EP relies on a single API "
                    "server and core client to coordinate scale up/down."
                )

        if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
            # engine args 中显式指定了 DP 配置。
            if self.distributed_executor_backend == "external_launcher":
                # external launcher 场景下，外部启动器负责全局 RANK，
                # 这里根据全局 RANK 自动推导 DP rank。
                self.data_parallel_rank = int(os.environ["RANK"]) // (
                    self.world_size // self.data_parallel_size
                )
                logger.info(
                    "Set data_parallel_rank to %d automatically.",
                    self.data_parallel_rank,
                )
            if not self.enable_elastic_ep:
                # 非 elastic EP 场景预分配几个 DP master 端口，后续按需 pop 使用。
                if not self._data_parallel_master_port_list:
                    self._data_parallel_master_port_list = get_open_ports_list(5)
                self.data_parallel_master_port = (
                    self._data_parallel_master_port_list.pop()
                )

            # data_parallel_rank 必须落在合法 DP rank 范围内。
            if not (0 <= self.data_parallel_rank < self.data_parallel_size):
                raise ValueError(
                    f"data_parallel_rank ({self.data_parallel_rank})"
                    f" must be in the range [0, {self.data_parallel_size})"
                )
        else:
            # 未在 engine args 中显式指定 DP 时，回退读取环境变量。
            # 典型场景是 offline SPMD。
            self.data_parallel_size = envs.VLLM_DP_SIZE
            self.data_parallel_rank = envs.VLLM_DP_RANK
            self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
            self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

            if self.data_parallel_size > 1 and self.is_moe_model is False:
                # dense 模型离线 DP 没有收益/不支持，提前拒绝。
                raise ValueError(
                    "Offline data parallel mode is not supported/useful"
                    " for dense models."
                )

        # data_parallel_index 默认与 data_parallel_rank 相同，部分 MoE 路径可能使用它。
        self.data_parallel_index = self.data_parallel_rank

        if self.distributed_executor_backend == "external_launcher":
            # external launcher 已经负责多进程管理，vLLM 内部不再启用 V1 multiprocessing。
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Disabling V1 multiprocessing for external launcher.")

        if self.distributed_executor_backend is None and self.world_size_across_dp > 1:
            # 如果 world_size 能放进当前节点，并且当前不在 Ray placement group 中，
            # 默认使用 multiprocessing。

            from vllm.v1.executor import ray_utils

            backend: DistributedExecutorBackend = "mp"
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD:
                # TPU SPMD 使用 uni backend。
                backend = "uni"
            elif current_platform.is_cuda() and self.nnodes > 1:
                # CUDA 多机显式使用 mp，由 nnodes/master_addr/master_port 组织拓扑。
                backend = "mp"
            elif (
                current_platform.is_cuda()
                and cuda_device_count_stateless() < self.world_size
            ):
                # 单机 CUDA 下，如果当前机器 GPU 数不足以容纳 world_size，
                # 用户必须显式选择 ray 或正确设置 nnodes。
                gpu_count = cuda_device_count_stateless()
                raise ValueError(
                    f"World size ({self.world_size}) is larger than the number of "
                    f"available GPUs ({gpu_count}) in this node. If this is "
                    "intentional and you are using:\n"
                    "- ray, set '--distributed-executor-backend ray'.\n"
                    "- multiprocessing, set '--nnodes' appropriately."
                )
            elif self.data_parallel_backend == "ray":
                # 用户显式要求 DP backend 用 ray，则分布式 executor 也使用 ray。
                logger.info(
                    "Using ray distributed inference because "
                    "data_parallel_backend is ray"
                )
                backend = "ray"
            elif ray_found:
                # 如果当前进程已经处于 Ray placement group 中，就继承 Ray 执行模式。
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized

                    if ray_is_initialized():
                        from ray.util import get_current_placement_group

                        if get_current_placement_group():
                            backend = "ray"
            self.distributed_executor_backend = backend
            logger.debug("Defaulting to use %s for distributed inference", backend)

        if self.distributed_executor_backend is None and self.world_size == 1:
            # 单 worker 场景使用 uni backend。
            self.distributed_executor_backend = "uni"

        if self.max_parallel_loading_workers is not None:
            # 当前该配置还没有实际支持，保留 warning 方便用户知道它被忽略。
            logger.warning(
                "max_parallel_loading_workers is currently "
                "not supported and will be ignored."
            )
        allowed_backends = ("mp", "uni", "external_launcher")
        if (
            self.distributed_executor_backend not in allowed_backends
            and self.nnodes > 1
        ):
            # nnodes > 1 只由 mp/uni/external_launcher 这些路径处理。
            raise ValueError(
                "nnodes > 1 can only be set when distributed executor "
                "backend is mp, uni or external_launcher."
            )

        if (
            self.all2all_backend in ("allgather_reducescatter", "naive")
            and self.eplb_config.use_async
        ):
            # 这些 all2all backend 与 async EPLB 组合已知可能 hang，因此强制回退同步 EPLB。
            logger.warning(
                "Async EPLB causes hangs with the '%s' all2all backend. "
                "Forcing synchronous EPLB.",
                self.all2all_backend,
            )
            self.eplb_config.use_async = False

    @property
    def use_ray(self) -> bool:
        return self.distributed_executor_backend == "ray" or (
            isinstance(self.distributed_executor_backend, type)
            and getattr(self.distributed_executor_backend, "uses_ray", False)
        )

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        # 延迟 import，避免配置模块与 executor 模块循环导入。
        from vllm.v1.executor import Executor

        # 如果请求 batch-invariant 模式，则关闭自定义 all-reduce，避免破坏不变性假设。
        if vllm_is_batch_invariant():
            self.disable_custom_all_reduce = True

        # distributed_executor_backend 允许是字符串、Executor 子类，或其 import path。
        if (
            self.distributed_executor_backend is not None
            and not isinstance(self.distributed_executor_backend, str)
            and not (
                isinstance(self.distributed_executor_backend, type)
                and issubclass(self.distributed_executor_backend, Executor)
            )
        ):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' 'uni', 'external_launcher', "
                " custom Executor subclass or its import path."
            )
        if self.use_ray:
            from vllm.v1.executor import ray_utils

            # 使用 Ray backend 前先确认 Ray 依赖可用。
            ray_utils.assert_ray_available()

        if not current_platform.use_custom_allreduce():
            # 当前平台不支持自定义 all-reduce 时，强制回退。
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on current platform."
            )
        if self.nnodes > 1:
            # 多机下自定义 all-reduce 不适用，统一禁用。
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce since we are running on multi-node."
            )
        if self.ray_workers_use_nsight and not self.use_ray:
            # nsight profiling 这个选项只对 Ray worker 路径有效。
            raise ValueError(
                "Unable to use nsight profiling unless workers run with Ray."
            )

        return self
