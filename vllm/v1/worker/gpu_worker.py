# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU Worker 实现。"""

import gc
import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from types import NoneType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
from vllm.distributed.eplb.eplb_utils import override_envs_for_eplb
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import (
    Handle,
    get_pp_group,
    get_tp_group,
)
from vllm.distributed.weight_transfer import WeightTransferEngineFactory
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.platforms import current_platform
from vllm.profiler.wrapper import CudaProfilerWrapper, TorchProfilerWrapper
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.utils import compute_iteration_details, report_usage_stats
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.worker.workspace import init_workspace_manager

from ...model_executor.model_loader import TensorizerLoader
from .gpu.warmup import warmup_kernels
from .utils import request_memory

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class AsyncIntermediateTensors(IntermediateTensors):
    """带惰性通信同步的 IntermediateTensors。"""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        comm_handles: list[Handle] | None = None,
        comm_postprocess: list[Callable[[], None]] | None = None,
    ) -> None:
        super().__init__(tensors)
        self._comm_handles = comm_handles
        self._comm_postprocess = comm_postprocess
        self._comm_waited = False

    def wait_for_comm(self) -> None:
        if self._comm_waited:
            return
        if self._comm_handles:
            for handle in self._comm_handles:
                handle.wait()
        if self._comm_postprocess:
            for fn in self._comm_postprocess:
                fn()
        self._comm_waited = True

    def __getattribute__(self, name: str):
        # 确保访问 `.tensors` 前通信已经完成。
        if name == "tensors" and not object.__getattribute__(self, "_comm_waited"):
            object.__getattribute__(self, "wait_for_comm")()
        return object.__getattribute__(self, name)


class Worker(WorkerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # 按 vLLM 环境变量配置 float32 matmul 精度。
        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION
        torch.set_float32_matmul_precision(precision)

        from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor

        self.elastic_ep_executor = ElasticEPScalingExecutor(self)

        # 进入 sleep 前备份的 buffer。
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # 权重传输引擎（按需初始化）。
        self.weight_transfer_engine = (
            WeightTransferEngineFactory.create_engine(
                self.vllm_config.weight_transfer_config,
                self.vllm_config.parallel_config,
            )
            if self.vllm_config.weight_transfer_config is not None
            else None
        )

        # Torch/CUDA profiler。是否启用与参数由 profiler_config 控制。
        # 在 profile(start=True) 时惰性创建，便于拿到完整命名信息。
        self.profiler: Any | None = None
        self.profiler_config = vllm_config.profiler_config

        # 这里只校验配置合法性，不提前实例化 profiler。
        if self.profiler_config.profiler not in ("torch", "cuda", None):
            raise ValueError(f"Unknown profiler type: {self.profiler_config.profiler}")

        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
        # 上一轮遗留的 PP 非阻塞发送句柄。
        self._pp_send_work: list[Handle] = []

    def sleep(self, level: int = 1) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # level=2 时会 offload 更多状态，这里先把 buffer 备份到 CPU。
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone() for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %s GiB memory, %s GiB memory is still in use.",
            format_gib(freed_bytes),
            format_gib(used_bytes),
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # 从 level=2 恢复时，把之前备份的 buffer 写回模型。
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

        # KV cache 被唤醒后，需要重置 cache_engine 内部状态，
        # 尤其是 FP8 的 scaling factor。
        if (
            (tags is None or "kv_cache" in tags)
            and self.cache_config.cache_dtype.startswith("fp8")
            and hasattr(self.model_runner, "init_fp8_kv_scales")
        ):
            self.model_runner.init_fp8_kv_scales()

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, (
                    "Sleep mode can only be used for one instance per process."
                )
            return allocator.use_memory_pool(tag=tag)
        else:
            return nullcontext()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    @instrument(span_name="Init device")
    def init_device(self):
        # 运行逻辑导读：
        # 1. 绑定当前进程到目标 CUDA 设备。
        # 2. 初始化分布式通信（先于显存快照，确保 NCCL 缓冲计入开销）。
        # 3. 采集初始化显存快照并计算“可申请显存”。
        # 4. 初始化 workspace 与 model runner。
        if self.device_config.device_type == "cuda":
            # Ray 会设置该环境变量，可能触发 graph 构建异常，先移除。
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            parallel_config = self.parallel_config
            if (
                parallel_config.distributed_executor_backend
                not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1
            ):
                # 优先使用本地 DP rank；缺失时回退到全局 DP rank。
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_index

                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size
                    * self.parallel_config.tensor_parallel_size
                )

                # 最终 local_rank = DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size
                assert self.local_rank < torch.cuda.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of bounds. "
                )
                visible_device_count = (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                )
                assert self.parallel_config.local_world_size <= visible_device_count, (
                    f"local_world_size ({self.parallel_config.local_world_size}) must "
                    f"be less than or equal to the number of visible devices "
                    f"({visible_device_count})."
                )

            self.device = torch.device(f"cuda:{self.local_rank}")
            current_platform.set_device(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # 先初始化分布式环境，再采显存快照。
            # 这样 NCCL 缓冲会先分配完，避免可用显存评估偏大。
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            if self.use_v2_model_runner:
                logger.info_once("Using V2 Model Runner", scope="local")

            # 设置随机种子。
            set_random_seed(self.model_config.seed)

            # NCCL 初始化后再采样显存。
            gc.collect()
            torch.cuda.empty_cache()

            # 记录当前显存快照，并计算目标可申请显存。
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
            self.requested_memory = request_memory(init_snapshot, self.cache_config)
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            logger.debug(
                "worker requested memory: %sGiB", format_gib(self.requested_memory)
            )
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # 初始化 workspace 管理器。
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # 构建 model runner（V1/V2 由环境开关决定）。
        if self.use_v2_model_runner:
            from vllm.v1.worker.gpu.model_runner import (
                GPUModelRunner as GPUModelRunnerV2,
            )

            # 临时类型兼容处理，避免静态类型报错。
            self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore
                self.vllm_config, self.device
            )
        else:
            from vllm.v1.worker.gpu_model_runner import (
                GPUModelRunner as GPUModelRunnerV1,
            )

            self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)

        if self.rank == 0:
            # 主 rank 在启用时上报使用统计。
            report_usage_stats(self.vllm_config)

    # FIXME(youkaichao & ywang96): 后续改为 TorchDispatchMode 拦截张量分配，
    # 取代当前 memory pool 方案。
    def load_model(self) -> None:
        dummy_weights = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
        if dummy_weights:
            (
                expanded_physical_to_logical,
                num_logical_experts,
                old_num_physical_experts,
            ) = self.elastic_ep_executor.receive_expert_mapping()
            num_physical_experts = expanded_physical_to_logical.shape[1]
            self.parallel_config.eplb_config.num_redundant_experts = (
                num_physical_experts - num_logical_experts
            )

        with (
            self._maybe_get_memory_pool_context(tag="weights"),
            set_current_vllm_config(self.vllm_config),
        ):
            self.model_runner.load_model(load_dummy_weights=dummy_weights)

        if dummy_weights:
            self.model_runner.setup_eplb_from_mapping(
                expanded_physical_to_logical, old_num_physical_experts
            )
            self.model_runner.eep_eplb_suppressed = True

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self, *args, **kwargs) -> None:
        self.model_runner.reload_weights(*args, **kwargs)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """探测模型峰值显存占用，估算可用于 KV cache 的安全显存。

        逻辑：
        1. 先做一次 profile run，拿到权重/激活/非 torch 开销。
        2. 再根据目标显存利用率计算 KV cache 可用字节数，尽量避免 OOM。

        提示：
            可通过 `gpu_memory_utilization` 控制 GPU 显存使用上限。
        """
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # 即便手动指定 KV cache 大小，仍需 profile run 触发编译路径
            # （例如 max_num_batched_tokens 对应内核）。
            self.model_runner.profile_run()

            msg = (
                f"Initial free memory {format_gib(self.init_snapshot.free_memory)} "
                f"GiB, reserved {format_gib(kv_cache_memory_bytes)} GiB memory for "
                "KV Cache as specified by kv_cache_memory_bytes config and "
                "skipped memory profiling. This does not respect the "
                "gpu_memory_utilization config. Only use kv_cache_memory_bytes "
                "config when you want manual control of KV cache memory "
                "size. If OOM'ed, check the difference of initial free "
                "memory between the current run and the previous run "
                "where kv_cache_memory_bytes is suggested and update it "
                "correspondingly."
            )
            logger.info(msg)
            return kv_cache_memory_bytes

        # 用 dummy 输入跑一次前向，统计模型真实显存开销。
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase

        free_gpu_memory = profile_result.after_profile.free_memory
        # 这里假设同卡其他进程在 profiling 期间没有显著变更显存占用。
        assert self.init_snapshot.free_memory >= free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {format_gib(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {format_gib(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )
        self.available_kv_cache_memory_bytes = (
            self.requested_memory - profile_result.non_kv_cache_memory
        )

        unrequested_memory = self.init_snapshot.free_memory - self.requested_memory
        logger.debug(
            "Initial free memory: %s GiB; Requested memory: %f (util), %s GiB",
            format_gib(self.init_snapshot.free_memory),
            self.cache_config.gpu_memory_utilization,
            format_gib(self.requested_memory),
        )
        logger.debug(
            "Free memory after profiling: %s GiB (total), %s GiB (within requested)",
            format_gib(free_gpu_memory),
            format_gib(free_gpu_memory - unrequested_memory),
        )
        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %s GiB",
            format_gib(self.available_kv_cache_memory_bytes),
            scope="local",
        )

        return int(self.available_kv_cache_memory_bytes)

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """如可用，返回当前 worker 的 KV connector 握手元数据。"""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # 对不需要跨 worker 交换握手元数据的 connector，返回 None。
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        tp_rank = get_tp_group().rank_in_group
        return {tp_rank: metadata}

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def update_max_model_len(self, max_model_len: int) -> None:
        """在自动适配 GPU 显存后，更新 max_model_len。

        当配置 `max_model_len=-1` 时，engine 会自动决定可容纳的最大上下文长度；
        worker 侧需要同步该值，保证执行路径一致。
        """
        self.model_config.max_model_len = max_model_len
        if self.model_runner is not None:
            self.model_runner.update_max_model_len(max_model_len)
        logger.debug("Updated max_model_len to %d", max_model_len)

    @instrument(span_name="Allocate KV cache")
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """按 kv_cache_config 分配 GPU KV cache。"""

        # KV connector 依赖 kv_cache_config，需在 initialize_kv_cache 前初始化。
        # 否则 initialize_kv_cache 可能注入与 connector 无关的 cache group
        # （如 cache sharing layer），导致 connector 视图不一致。
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            with allocator.use_memory_pool(tag="kv_cache"):
                self.model_runner.initialize_kv_cache(kv_cache_config)
        else:
            self.model_runner.initialize_kv_cache(kv_cache_config)

    @instrument(span_name="Warmup (GPU)")
    def compile_or_warm_up_model(self) -> float:
        # 运行逻辑导读：
        # 1. 计算需要编译/预热的 batch size 集合。
        # 2. 通过 dummy run 触发编译与内核选择。
        # 3. 可选执行 CUDA Graph capture。
        # 4. 预热采样相关路径并恢复随机种子。
        warmup_sizes = []

        if self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            # 对不在 cudagraph capture 列表中的 size 也做预热编译，
            # 以覆盖用户关心的性能点（如 chunked prefill 的大 batch）。
            compile_sizes = self.vllm_config.compilation_config.compile_sizes
            warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []
            cg_capture_sizes: list[int] = []

            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            # 对每个 compile_range：若 warmup/capture 都未覆盖该区间，
            # 则补上区间末端 size，保证至少有一次编译/预热命中该范围。
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        # 预热阶段跳过 EPLB，避免把 dummy run 计入真实指标。
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # 在 CUDA Graph capture 前先做 kernel warmup/tuning。
        kernel_warmup(self)

        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(
            self, "peak_activation_memory"
        ):
            # 给出更精细的 KV cache 配置建议。
            # memory_profiling 不包含 CUDAGraph 占用，可能高估可用空间；
            # 这里把权重/激活/非 torch/CUDAGraph 合并后再估算。
            # 经验上 profiling 略有低估，因此额外预留 150MiB 缓冲。
            redundancy_buffer_memory = 150 * (1 << 20)
            non_kv_cache_memory = (
                self.model_runner.model_memory_usage
                + self.peak_activation_memory
                + self.non_torch_memory
                + cuda_graph_memory_bytes
            )
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.requested_memory)
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )

            msg = (
                f"Free memory on device "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_memory)} GiB). "
                f"Actual usage is {format_gib(self.model_runner.model_memory_usage)} "
                f"GiB for weight, {format_gib(self.peak_activation_memory)} GiB "
                f"for peak activation, {format_gib(self.non_torch_memory)} GiB "
                f"for non-torch memory, and {format_gib(cuda_graph_memory_bytes)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"config with `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_requested_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into requested memory, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize gpu memory. Current kv cache memory in use is "
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.debug(msg)

        if self.use_v2_model_runner:
            # V2：通过完整 execute_model + sample_tokens 路径触发 Triton JIT。
            warmup_kernels(self.model_runner)
        elif get_pp_group().is_last_rank:
            # V1：预热 sampler，并按最大形状预分配 logits/采样相关缓冲，
            # 降低显存碎片风险。
            # 这里故意放在 capture_model 之后，避免被 empty_cache 清掉。
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # 同样跳过 EPLB，避免 dummy 指标污染。
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        # 重置随机种子，避免初始化/预热阶段影响后续随机行为。
        set_random_seed(self.model_config.seed)

        return self.compilation_config.compilation_time

    def reset_mm_cache(self) -> None:
        self.model_runner.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        self.model_runner.reset_encoder_cache()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """获取 model runner 采集的 encoder 耗时统计。"""
        return self.model_runner.get_encoder_timing_stats()

    def annotate_profile(self, scheduler_output):
        # 添加 trace 注释，便于区分每轮 context/generation 请求与 token 数。
        # context 请求指尚未产生输出 token 的请求。
        if not self.profiler:
            return nullcontext()

        self.profiler.step()

        iteration_details = compute_iteration_details(scheduler_output)

        annotation = "".join(
            [
                "execute_context_",
                str(iteration_details.num_ctx_requests),
                "(",
                str(iteration_details.num_ctx_tokens),
                ")_generation_",
                str(iteration_details.num_generation_requests),
                "(",
                str(iteration_details.num_generation_tokens),
                ")",
            ]
        )
        return self.profiler.annotate_context_manager(annotation)

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    @torch.inference_mode()
    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # 运行逻辑导读：
        # 1. 等待上轮 PP 异步发送完成，避免跨轮次资源冲突。
        # 2. 非首 stage 时先异步接收上游中间张量。
        # 3. 调用 model_runner.execute_model 执行当前轮前向。
        # 4. 非末 stage 时异步发送中间张量给下游；末 stage 返回采样输出。

        # 确保上一轮未完成的 PP 非阻塞发送已结束。
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        all_gather_tensors = {}
        compilation_config = self.vllm_config.compilation_config
        parallel_config = self.vllm_config.parallel_config

        if (
            parallel_config.pipeline_parallel_size > 1
            and compilation_config.pass_config.enable_sp
            and forward_pass
        ):
            # 当前仅 V1 GPUModelRunner 支持该路径。
            assert not self.use_v2_model_runner
            num_scheduled_tokens_np = np.array(
                list(scheduler_output.num_scheduled_tokens.values()),
                dtype=np.int32,
            )
            # TODO(lucas): 这里会与 execute_model 内部重复调用一次
            # `_determine_batch_execution_and_padding`。理想情况应只调用一次，
            # 但需要较大规模 PP 重构。
            _, batch_desc, _, _, _ = (
                self.model_runner._determine_batch_execution_and_padding(
                    num_tokens=num_scheduled_tokens,
                    num_reqs=len(num_scheduled_tokens_np),
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=num_scheduled_tokens_np.max(),
                    use_cascade_attn=False,  # TODO(lucas): 补齐 cascade attention
                )
            )
            all_gather_tensors = {
                "residual": not is_residual_scattered_for_sp(
                    self.vllm_config, batch_desc.num_tokens
                )
            }

        if forward_pass and not get_pp_group().is_first_rank:
            tensor_dict, comm_handles, comm_postprocess = (
                get_pp_group().irecv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
            )
            assert tensor_dict is not None
            intermediate_tensors = AsyncIntermediateTensors(
                tensor_dict,
                comm_handles=comm_handles,
                comm_postprocess=comm_postprocess,
            )

        with self.annotate_profile(scheduler_output):
            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            if (
                self.use_v2_model_runner
                and self.model_runner.is_pooling_model
                and output is None
            ):
                output = self.model_runner.pool()  # type: ignore
            if isinstance(
                output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType
            ):
                return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert (
            parallel_config.distributed_executor_backend != "external_launcher"
            and not get_pp_group().is_last_rank
        )

        # 启动中间张量的 PP 非阻塞发送（下一轮开头统一 wait）。
        self._pp_send_work = get_pp_group().isend_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )

        return None

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # 检查是否启用 profiling。
        if self.profiler_config is None or self.profiler_config.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )

        if is_start:
            # 组合前缀与 rank 后缀，生成 trace 名称。
            from vllm.distributed.utils import get_worker_rank_suffix

            rank_suffix = get_worker_rank_suffix(global_rank=self.rank)

            # 拼接完整 trace 名称。
            if profile_prefix:
                trace_name = f"{profile_prefix}_{rank_suffix}"
            else:
                trace_name = rank_suffix

            # 仅在第一次 start 时创建 profiler wrapper。
            if self.profiler is None:
                profiler_type = self.profiler_config.profiler
                if profiler_type == "torch":
                    self.profiler = TorchProfilerWrapper(
                        self.profiler_config,
                        worker_name=trace_name,
                        local_rank=self.local_rank,
                        activities=["CPU", "CUDA"],
                    )
                    logger.debug(
                        "Starting torch profiler with trace name: %s", trace_name
                    )
                elif profiler_type == "cuda":
                    self.profiler = CudaProfilerWrapper(self.profiler_config)
                    logger.debug("Starting CUDA profiler")
                else:
                    # 理论上配置校验已覆盖该分支。
                    raise ValueError(
                        f"Invalid profiler value of {self.profiler_config.profiler}"
                    )

            # 若已初始化，则仅重启采集，保持首次创建时的 trace 名称。
            self.profiler.start()
        else:
            if self.profiler is None:
                logger.warning("Profiler was not started, nothing to stop.")
                return
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1, uniform_decode=True)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # 只要进程存活，worker 就视为健康。
        return

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader

        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(self, tensorizer_config: "TensorizerConfig") -> None:
        TensorizerLoader.save_model(
            self.get_model(),
            tensorizer_config=tensorizer_config,
            model_config=self.model_config,
        )

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        """初始化权重传输机制。

        对 NCCL 后端，会与 trainer 建立专用进程组。

        参数：
            init_info: 各后端需要的初始化信息字典。
        """
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. "
                "Please set weight_transfer_config to enable weight transfer."
            )
        # 先将通用 dict 解析为后端特定的强类型配置。
        typed_init_info = self.weight_transfer_engine.parse_init_info(init_info)
        self.weight_transfer_engine.init_transfer_engine(typed_init_info)

    def update_weights(self, update_info: dict) -> None:
        """接收 trainer 的批量权重更新。 

        参数：
            update_info: 各后端需要的权重更新信息字典。
        """
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. "
                "Please set weight_transfer_config to enable weight transfer."
            )

        # 先把通用 dict 解析为后端特定的强类型更新配置。
        typed_update_info = self.weight_transfer_engine.parse_update_info(update_info)

        model = self.model_runner.model

        if typed_update_info.is_checkpoint_format:
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
                initialize_layerwise_reload,
            )

            # checkpoint 格式权重走分层 reload 路径。
            with torch.device(self.device):
                initialize_layerwise_reload(model)
                self.weight_transfer_engine.receive_weights(
                    typed_update_info,
                    load_weights=model.load_weights,
                )
                finalize_layerwise_reload(model, self.model_config)
        else:
            # 权重已是 kernel 格式，直接拷贝到参数张量。
            def load_weights_direct(
                weights: list[tuple[str, torch.Tensor]],
            ) -> None:
                for name, weight in weights:
                    param = model.get_parameter(name)
                    param.copy_(weight)

            self.weight_transfer_engine.receive_weights(
                typed_update_info,
                load_weights=load_weights_direct,
            )

    def shutdown(self) -> None:
        # 解释器退出阶段，has_kv_transfer_group 可能已经不可用。
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()
        if self.profiler is not None:
            self.profiler.shutdown()

        if weight_transfer_engine := getattr(self, "weight_transfer_engine", None):
            weight_transfer_engine.shutdown()

    def elastic_ep_execute(self, execute_method: str, *args, **kwargs):
        return self.elastic_ep_executor.execute(execute_method, *args, **kwargs)


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """初始化 worker 的分布式环境。"""
    attention_config = vllm_config.attention_config
    parallel_config = vllm_config.parallel_config
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    init_batch_invariance(attention_config.backend)
    override_envs_for_eplb(parallel_config)
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_method = distributed_init_method or "env://"
    init_distributed_environment(
        parallel_config.world_size, rank, init_method, local_rank, backend
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )

    # 在 KV cache 初始化前先初始化 ec connector。
    # 注意：EPD 解耦模式下，Encoder-only 实例不初始化 KV cache。
    ensure_ec_transfer_initialized(vllm_config)
