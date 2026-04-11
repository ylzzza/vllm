# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsReader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set 控制是否在 update_from_outputs() 返回的
        # EngineCoreOutputs 中额外携带一个“已结束请求 ID 集合”。
        # 目前主要用于多引擎场景，便于高效追踪请求生命周期。
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        self.prev_step_scheduled_req_ids: set[str] = set()

        # 调度约束。
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # 为 Scheduler 创建 KVConnector。
        # 注意：每个 Worker 也会有自己的 KVConnector（Role=WORKER）。
        # KVConnector 用于远端 KV 的推送/拉取（P/D 解耦与卸载场景）。
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        if self.vllm_config.kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = (
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # 请求映射：req_id -> Request
        self.requests: dict[str, Request] = {}
        # 调度策略
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # 请求优先队列。
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # 记录“上一步到这一步之间”结束的请求 ID。
        # 用于通知 worker 释放这些请求的缓存状态。
        # 每次调度步结束后会清空。
        self.finished_req_ids: set[str] = set()

        # 等待流式输入的请求计数，用于计算未完成请求数。
        self.num_waiting_for_streaming_input: int = 0

        # KVConnector：异步加载/接收 KV 中的请求集合。
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder 相关。
        # 如适用，计算 encoder cache 大小。
        self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        self.mm_budget = mm_budget = (
            MultiModalBudget(vllm_config, mm_registry)
            if self.supports_mm_inputs
            else None
        )

        # 注意：纯文本 encoder-decoder 模型为了工程统一，
        # 在实现上也走多模态接口。
        # 示例：https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens

        # 创建 KV cache 管理器。
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            hash_block_size=self.block_size,
            metrics_collector=self.kv_metrics_collector,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER

        def has_mamba_layers(kv_cache_config: KVCacheConfig) -> bool:
            return any(
                isinstance(group_spec.kv_cache_spec, MambaSpec)
                for group_spec in kv_cache_config.kv_cache_groups
            )

        self.has_mamba_layers = has_mamba_layers(kv_cache_config)
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        if self.vllm_config.model_config.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_reader = RoutedExpertsReader.create()

            assert len(kv_cache_config.kv_cache_groups) > 0, (
                "enable_return_routed_experts requires at least one kv cache group"
            )
            self.max_num_kv_tokens = (
                kv_cache_config.num_blocks // len(kv_cache_config.kv_cache_groups) + 1
            ) * self.block_size

            self.routed_experts_reader.attach_buffer(
                max_num_kv_tokens=self.max_num_kv_tokens,
                vllm_config=self.vllm_config,
            )

        self._pause_state: PauseState = PauseState.UNPAUSED

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        assert num_external_computed_tokens == 0, (
            "External KV connector is not verified yet"
        )
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # 在 prefill 阶段按 block 对齐切分，包含：
        # 1. 非恢复请求：num_computed_tokens < num_prompt_tokens
        # 2. 恢复请求：num_computed_tokens < (num_prompt_tokens + num_output_tokens)
        # 注意：用 `request.num_tokens - 1` 来跳过常规 decode 判定边界。
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # 为了让 Mamba 状态按 block 对齐缓存，`num_new_tokens`
            # 需要是 `block_size` 的倍数。
            # 例外：若小于 `block_size`，该段状态不缓存，无需特殊处理。
            # 另外在 Eagle 模式下，FullAttn 会裁掉最后一个匹配块；
            # 为避免 Mamba cache miss，最后一段不能小于 `block_size`。
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            # Eagle 裁剪
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # 对齐到 block_size
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # 强制缓存最后一段
                num_new_tokens = last_cache_position - num_computed_tokens
            else:
                # 预填充最后几个 token
                pass
        return num_new_tokens

    def schedule(self) -> SchedulerOutput:
        # 调度核心逻辑（运行路径）：
        # 1. 先调度 running 队列：在 token/encoder/KV 预算内尽可能推进。
        # 2. 再调度 waiting 队列：将可运行请求转入 running。
        # 3. 如遇 KV 不足，按策略抢占低优先级请求并回收资源。
        # 4. 汇总本步输出（新请求、缓存请求、spec token、encoder 输入等）。
        #
        # 说明：这里不硬分 “prefill phase / decode phase”。
        # 每个请求只有两个关键计数：
        # - num_computed_tokens：已计算到的位置
        # - num_tokens_with_spec：目标位置（prompt + output + spec）
        # 每步都尝试让前者追上后者，因此天然兼容 chunked prefill、
        # prefix cache、spec decoding 及后续 jump decoding 优化。

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # 全暂停时，不调度任何请求。
            token_budget = 0

        # Encoder 相关。
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode 相关。
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # 日志时间戳。
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # 第一阶段：调度 RUNNING 队列。
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                # 等价于：(num_computed_tokens + 1) - (num_output_placeholders - 1)。
                # 由于 computed 计数里也包含 output placeholders，
                # 这里减去 (num_output_placeholders - 1) 以剔除 draft token 影响，
                # 从而在“全部 draft 被拒绝”时也能正确判断无需继续调度。
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # 异步调度：若已确定上一轮达到 request.max_tokens，
                # 则避免多调度一步。这里不调度“部分 draft token”，
                # 以免破坏统一 decode 优化路径。
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # 确保输入位置不超过模型最大长度（spec decoding 下尤为必要）。
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            # 调度 encoder 输入。
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                # 该请求本轮不可调度，常见原因：
                # 1. 没有可调度新 token（例如 PP>1 时 prompt 已下发但尚未完成，
                #    或异步调度下已达到 max_total_tokens/max_model_len）。
                # 2. encoder 计算预算耗尽。
                # 3. encoder cache 耗尽。
                # 4. mamba_cache_mode="align" 下，不足以切出 block 对齐分片。
                # 注意：这里用 continue 而不是 break，
                # 因此不会严格遵循 FCFS，可让更低优先级请求获得调度机会。
                req_index += 1
                continue

            # 为该请求分配本轮新需要的 KV block。
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # 可调度，成功分配到 block。
                        break

                    # 不可调度：触发抢占，回收低优先级请求资源。
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # 若被抢占请求本轮已占用 encoder 预算，则归还预算。
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # 已无可抢占请求，当前请求无法调度。
                        break

            if new_blocks is None:
                # 当前请求无法调度。
                break

            # 正式记录该请求的调度结果。
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode 相关处理。
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # 下一步开始前会在 `update_draft_token_ids` 中写入新的 spec tokens。
                request.spec_token_ids = []

            # Encoder 相关。
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # 分配 encoder cache。
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # 记录本轮 RUNNING 队列里涉及到的 LoRA。
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # 第二阶段：调度 WAITING 队列。
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            # 临时队列：收集本轮需要跳过的请求，稍后放回 waiting 头部。
            skipped_waiting_requests = create_request_queue(self.policy)

            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()
                request_id = request.request_id

                # KVTransfer：若仍在等待远端 KV，先跳过。
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        if request.num_preemptions:
                            # 说明这是“被抢占后恢复”的加载，不是新请求。
                            request.status = RequestStatus.PREEMPTED
                        else:
                            request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # 结构化输出若仍在等待 FSM 编译，先跳过该请求。
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Streaming：若还在等下一段流式输入，先跳过。
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    assert not request.streaming_queue
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                # 检查加入后是否仍满足 max_loras 约束。
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # 超出 max_loras，跳过。
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0

                # 获取已缓存 token。
                if request.num_computed_tokens == 0:
                    # 先查本地缓存。
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    # 若使用 KVConnector，再查外部缓存。
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # KVConnector 无法确定命中 token 数，当前请求不可调度。
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # 总已计算 token（本地 + 外部）。
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                else:
                    # KVTransfer：异步接收完成后，WAITING 请求的
                    # num_computed_tokens 可能 > 0。
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer：远端 KV 正在加载，不为新计算分配资源。
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # 计算本轮可调度 token 数。
                    # 这里用 `request.num_tokens` 而不是 `request.num_prompt_tokens`，
                    # 以兼容包含 output token 的恢复请求。
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # 只有显式开启 chunked prefill，pooling 请求才允许分块。
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # 未开启 chunked_prefill 时，无法继续切分调度，直接退出本轮。
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # 调度 encoder 输入。
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # 该请求不可调度。
                            break

                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                # 处理一个边界情况：P/D 解耦 + Spec Decoding 时，
                # 可能多分配一个 block，导致本地/远端 block 数不一致。
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                # 判断是否需要分配 cross-attention blocks。
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # 该请求不可调度。

                    # 注意：需要从 encoder cache manager 中撤销本请求的触碰状态。
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer：把分配信息同步给 connector，
                # connector 会据此判断该请求是否需要执行远端加载。
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                # 请求通常已从 waiting 弹出；
                # 只有上面 new_blocks is None 时才会被重新放回。
                request = self.waiting.pop_request()
                if load_kv_async:
                    # 异步加载时：先占位分配内存，并切到 WAITING_FOR_REMOTE_KV。
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # 记录 prefix cache 命中 token 数。
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder 相关。
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # 分配 encoder cache。
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # 为外部加载的 encoder 输入占位分配 cache。
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            # 把本轮跳过的请求放回 waiting 队列头部。
            if skipped_waiting_requests:
                self.waiting.prepend_requests(skipped_waiting_requests)

        # 调度后进行约束校验。
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # RUNNING 队列中部分请求本轮可能未被调度，
        # 因此“本轮调度请求数”可以小于 len(self.running)。
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # 计算 running 队列请求的最长公共前缀块，可用于 cascade attention 优化。
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # 构造调度输出。
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # 记录本轮被调度请求 ID。
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids 是 scheduler 内部已有状态，
            # 并非“本轮新调度”产生的数据。
            # 它表示“上一步到这一步之间”结束的请求 ID。
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )

        # 说明：build_connector_meta 同时承担三件事：
        # 1. 规划 KV cache 的 store 行为
        # 2. 将 KV load/save 操作封装成透明元信息对象
        # 3. 清理 connector 的内部临时状态
        if self.connector is not None:
            meta: KVConnectorMetadata = self.connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.kv_connector_metadata = meta

        # 为 ECConnector 构建 connector meta。
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """抢占一个请求，并把它放回 waiting 队列。

        注意：调用方需要先在 running 队列外部把该请求移除。
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # 将请求放回 waiting 队列头部，等待后续恢复。
        self.waiting.prepend_request(request)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # 在“请求已被调度”之后再推进 num_computed_tokens：
        # 1. 当前步 scheduler_output 需要保留原始 scheduled token 数，
        #    以便正确构造输入 ID。
        # 2. 在这里推进计数，可让 prefill 请求在下一调度步立即继续推进。
        # 3. 若后续出现 token（如 spec token）被拒绝，会在 update_from_output
        #    里回调修正 num_computed_tokens。
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )

            # _free_encoder_inputs 依赖 num_computed_tokens。
            # 虽然 speculative decoding 可能在 _update_from_output 再次修正该值，
            # 但这里调用依然安全：encoder 输入始终属于 prompt，不受 spec token
            # 接受/拒绝影响。
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # 清空 finished request IDs。
        # 不能直接 clear 原集合，否则会影响已返回的 scheduler_output 视图。
        self.finished_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """用下一段流式输入更新等待中的会话请求。

        会丢弃上一段输入尾部“最后一个采样输出 token”。
        """

        # 当前流式输入行为：只保留已计算输出 token（丢弃最后一个采样 token）。
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # 把保留输出 token 回填到 prompt。
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # 为新增 token 更新 block 哈希。
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # 注意：在 PP+异步调度下，token id 通过 GPU 直接广播通路
            # （`input_batch.prev_sampled_token_ids`）消费，这里可省略该载荷。
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # 使用 PP 时，首末 stage worker 之间没有直连通信，
                # 因此 scheduler 需要把采样 token 回传；非该场景下 model runner
                # 会自行缓存，无需额外回传。
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids
            if idx >= num_running_reqs:
                assert not scheduled_in_prev_step
                resumed_req_ids.add(req_id)
            if not scheduled_in_prev_step:
                all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """决定当前步需要调度哪些 encoder 输入，并同步更新预算。

        返回值：
        1. 本轮要调度的 encoder 输入索引列表
        2. 调整后的 num_new_tokens
        3. 调整后的 encoder_compute_budget
        4. 需要走外部 encoder cache 加载的输入索引列表

        只有同时满足以下条件，某个 encoder 输入才会被调度：
        1. 与当前计算区间 [num_computed_tokens, num_computed_tokens + num_new_tokens)
           存在重叠。
        2. 尚未在本地 encoder cache 中命中。
        3. 远端 encoder cache（ECConnector）也未直接可用，或需外部加载。
        4. encoder token 预算足够。
        5. encoder cache 有可分配空间。

        若受预算或缓存限制无法调度 encoder 输入，会回退 num_new_tokens，
        仅调度到该 encoder 输入起点之前的 decoder token。

        注意：num_computed_tokens 同时包含本地缓存和外部缓存命中。
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # 调度器按 request 粒度工作（一个 request 可能含多个 encoder 输入），
        # 因此这里使用临时变量在“单个 encoder 输入粒度”上做预算核算。
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            # 两个区间重叠时，说明本轮需要对应 encoder 输出：
            # [num_computed_tokens, num_computed_tokens + num_new_tokens)
            # [start_pos, start_pos + num_encoder_tokens)
            if (
                start_pos
                >= num_computed_tokens + num_new_tokens + shift_computed_tokens
            ):
                # 本轮还不需要该 encoder 输入。
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # encoder-decoder 模型下，start_pos 应始终为 0。
                # 一旦 num_computed_tokens > 0，表示 decoder 已开始，
                # 也就意味着 encoder 输入已在此前计算过，可直接跳过。
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # 该 encoder 输入已计算并写入 decoder KV cache。
                continue

            if not self.is_encoder_decoder:
                # 当前 encoder-decoder 路径尚未使用这里这套 encoder cache 复用逻辑。
                if item_identifier in mm_hashes_to_schedule:
                    # 本轮已安排过相同输入，避免重复调度。
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # 历史步骤已缓存，直接复用。
                    continue

            # 若禁用 encoder 输入分块，则不能只调度一个多模态项的部分区间。
            # 若当前范围只能覆盖其一部分，则回退到该项起点之前。
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # 回退时考虑 EAGLE 的 shift，避免 encoder cache miss。
                # 目标是确保调度区间在 shift 后仍停在 start_pos 之前。
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # encoder cache 满或 encoder 预算耗尽。
                # 这里假设 encoder 输入应整体处理（通常是双向注意力）。
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # 仅调度到 encoder 输入起点前的 decoder token。
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # prefix cache 可能导致 num_computed_tokens 已超过 start_pos，
                    # 但 encoder 输入仍不可用，此时本轮该请求无法调度任何 token。
                    num_new_tokens = 0
                break

            # 计算当前调度区间内需要的 encoder embedding 数量。
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # 当前占位区间不含任何 embedding，跳过该 encoder 输入。
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # 收集本轮已调度且启用结构化输出的请求 ID。
        # bitmask 的行顺序与此列表一致。
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # 这些 block 对应“外部已计算但加载失败”的 token。
            # 需要找出受影响请求，并回退其 computed 计数以触发重算。
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids
            )

        # 注意：num_scheduled_tokens 规模可达 1K+，下面循环是热点路径，
        # 应尽量避免在循环内做重操作。
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # 跳过 KV 加载失败后被标记失败/重调度的请求。
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # 请求已经结束。可能是执行过程中被外部中止（例如 PP 或异步调度）。
                # 注意：delay_free_blocks=True 时（KV connector 异步传输），
                # 被中止请求可能暂不从 requests 中删除，因此这里以 is_finished()
                # 作为最终判定。
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens 表示本轮实际生效的已处理 token 数。
                # 若 draft token 有拒绝，需要按拒绝数回退该计数。
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # 异步调度下，num_output_placeholders 也包含本轮 spec token 占位，
                # 因此同样按拒绝数回退。
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status

            # 检查停止条件并更新请求状态。
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling 请求一旦有输出即可结束。
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            routed_experts = None
            finish_reason = None
            if stopped:
                routed_experts = self._get_routed_experts(request)

                # 先抓取 finish_reason，再调用 _handle_stopped_request。
                # 后者可能把可续流请求状态改回 WAITING。
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # 按需提取采样 logprobs。
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                ok = struct_output_request.grammar.accept_tokens(req_id, new_token_ids)
                if not ok:
                    logger.warning(
                        "Unexpected: grammar rejected tokens %s for request %s.",
                        new_token_ids,
                        req_id,
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # 获取该请求的 prompt logprobs。
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if (
                new_token_ids
                or pooler_output is not None
                or kv_transfer_params
                or stopped
            ):
                # 组装该请求的 EngineCoreOutput。
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        num_external_computed_tokens=request.num_external_computed_tokens,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # 不变量：EngineCore 不返回“部分 prefill”输出。
                assert not prompt_logprobs_tensors

        # 把已停止请求从 running / waiting 队列移除。
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # 该路径较少发生，对性能影响很小。
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        # KVConnector：更新已完成 KV 传输请求状态。
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # 收集 KV cache manager 侧事件。
        events = self.kv_cache_manager.take_events()

        # 收集 connector 侧事件。
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # 发布本轮收集到的 KV cache 事件。
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # 为本轮有输出的各 client 构建 EngineCoreOutputs。
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # 附带“自上次发送输出以来”结束的请求 ID。
            for client_index, finished_set in finished_req_ids.items():
                # 为该 client 写入 finished request 集合。
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # 统计仅回传给一个前端实例。
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # 即使本轮没有请求输出，也要返回统计信息。
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _handle_stopped_request(self, request: Request) -> bool:
        """返回请求是否最终结束（可恢复请求可能返回 False）。"""
        if not request.resumable:
            return True

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # 流式请求已结束。
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self.waiting.add_request(request)
        return False

    def _get_routed_experts(self, request: Request) -> np.ndarray | None:
        if not self.vllm_config.model_config.enable_return_routed_experts:
            return None

        kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
        block_ids = kv_blocks.get_block_ids()[0]
        num_tokens = request.num_tokens - 1

        # 计算 slot 映射
        block_ids_array = np.array(block_ids, dtype=np.int32)
        num_blocks = len(block_ids)
        block_size = self.block_size

        # 生成 block 内偏移
        block_offsets = np.arange(0, block_size)

        # 计算 slot 映射：slot = block_id * block_size + offset
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_array.reshape((num_blocks, 1)) * block_size
        ).flatten()[:num_tokens]

        return self.routed_experts_reader.get_routed_experts(indices=slot_mapping)

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # 追加新生成 token 并检查停止条件。
        # 若请求仍处于 prefill，model runner 预期返回空 token 列表。
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # 检查停止并更新状态。
            # 必须在构造 EngineCoreOutput 前完成。
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # 需要时裁剪尾部 token。
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # 优化：空集合时避免无意义的 list(set)。
        if not cached_encoder_input_ids:
            return

        # 用 list(set) 防止迭代过程中原集合被修改。
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # 以 Whisper 为例：只要生成了一个 token，
                # 就说明 encoder 输入已完成，Cross-Attention KV 已就绪。
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # encoder 输出已被消费，并写入 decoder KV cache。
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # 请求可能已经结束，跳过。
                continue

            if request.is_prefill_chunk:
                # prefill 分块阶段忽略 draft tokens。
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # 写入新生成的 spec token ids。
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # 请求可能已经结束，跳过。
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # 将 draft 裁剪到本轮已调度的 spec token 数（例如 chunked prefill）。
            del spec_token_ids[orig_num_spec_tokens:]
            # 过滤不符合 grammar 的 spec token。
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # 补齐到原始 spec token 数量。
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """返回 (num_running_reqs, num_waiting_reqs)。"""
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # 入队下一段输入（或结束哨兵）。
                existing.streaming_queue.append(update)
            elif update is not None:
                # 启动下一段输入。
                self._update_request_as_session(existing, update)
            else:
                # 流式输入会话结束。
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            self.waiting.add_request(request)
            self.requests[request.request_id] = request
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        """处理来自调度器外部的结束信号。

        例如：客户端断连时，API server 可主动中止请求。
        若 request_ids 为 None，表示结束所有请求。

        返回：
            被本次处理终止的请求 (req_id, client_index) 列表，
            已经结束的请求不会重复返回。
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # 第一轮：收集需要从队列移除的请求。
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # 无效请求 ID。
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # 为提升效率，批量从队列移除。
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # 第二轮：更新状态并释放资源。
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return [(r.request_id, r.client_index) for r in valid_requests]

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()

        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = len(self.waiting) - self.num_waiting_for_streaming_input
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """重置 KV prefix cache。

        reset_running_requests=True 时，会先抢占全部 running 请求并转回 waiting；
        否则仅在没有 running 请求占用 KV cache 时才执行重置。
        """
        if reset_running_requests:
            # 记录日志时间戳。
            timestamp = time.monotonic()
            # 通过将所有 running 请求抢占回 waiting，使现有 KV 全部失效。
            # 这样可将 block 引用计数降到 0，确保重置成功。
            # 按逆序抢占，后续恢复时可保持 FIFO 顺序。
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp)
                # 异步调度下丢弃最新输出占位，避免重复 token。
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True

            # 清空“上一步已调度请求”缓存。因为这里在同一步强制执行
            # preempt + resume，需要让 model runner 把它们当作未在上步调度过。
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            logger.warning("reset_connector called but no KV connector is configured.")
            return False

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """重置 encoder cache，使所有缓存的 encoder 输出失效。

        当模型权重更新后应调用，避免复用过期的视觉 embedding。
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            encoder_cache_usage=self._get_encoder_cache_usage(),
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def _get_encoder_cache_usage(self) -> float:
        """获取 encoder cache 使用率（0.0 到 1.0）。"""
        ecm = self.encoder_cache_manager
        if ecm.cache_size == 0:
            return 0.0
        used_slots = ecm.cache_size - ecm.num_free_slots
        return used_slots / ecm.cache_size

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KVConnector 相关方法
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """必要时调用 KV connector 的 request_finished 逻辑。

        返回可选的 kv transfer 参数，写入请求输出。
        """
        if self.connector is None:
            return False, None

        # 把 block 表交给 connector 之前，先释放窗口外 prefix blocks。
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # 后续在所有 connector 强制支持 HMA 后，这条路径可移除。
            # 当前理论上已关闭 hybrid memory allocator，这里再做一次兜底校验。
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """KVConnector：检查请求是否已完成远端 KV 接收。

        finished_recving_kv_req_ids 在上一步 update_from_output 中由 worker
        侧 connector 回传并更新。
        接收完成后会缓存相应 block，并把请求状态从 WAITING_FOR_REMOTE_KV
        恢复为 WAITING。
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        if request.request_id in self.failed_recving_kv_req_ids:
            # 请求发生过 KV 加载失败；num_computed_tokens 已在
            # _update_requests_with_invalid_blocks 中更新。
            if request.num_computed_tokens:
                # 缓存仍有效的已计算 token。
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # 没有有效 token，释放已分配 block（重试时可能有本地命中）。
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # block 已就绪，正式写入缓存。
            (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
            num_computed_tokens = len(block_ids) * self.block_size
            # 处理“请求 token 数不足一个 block”的情况。
            num_computed_tokens = min(num_computed_tokens, request.num_tokens)
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            # 仅在启用缓存时执行 cache_blocks。
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

            # 更新请求状态，允许后续继续调度。
            request.num_computed_tokens = num_computed_tokens

        # 标记已就绪。
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """KVConnector：基于输出更新调度器状态。

        Worker 侧 connector 会在输出中带回 finished_recving / finished_sending。
        1. finished_sending：可释放对应 blocks。
        2. finished_recving：标记为下一步可恢复调度。
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KVConnector：更新上一步的接收/发送完成状态。
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """识别并更新受无效 KV block 影响的请求。

        会扫描给定请求集合，找出命中无效 block 的请求，并将其
        `num_computed_tokens` 回退到最长有效前缀；同时统计需要重算的 token 数。

        参数：
            requests: 待扫描请求集合。
            invalid_block_ids: 无效 block ID 集合。
            evict_blocks: 是否收集待驱逐 block（异步请求尚未缓存，通常为 False）。

        返回：
            (affected_req_ids, total_affected_tokens, blocks_to_evict)
            1. affected_req_ids: 受影响请求 ID 集合。
            2. total_affected_tokens: 所有受影响请求需重算的 token 总数。
            3. blocks_to_evict: 需从缓存驱逐的 block（含下游依赖）。
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # 若一个无效 block 被 batch 内多个请求共享，这些请求都需重调度；
        # 但只应由第一个请求负责重算。该集合用于追踪“已标记重算”的 block。
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO(davidb): 增加对 hybrid memory allocator 的支持
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # 仅遍历可能包含“外部计算 token”的 block。
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # 异步加载：若 num_computed_tokens 已设置，说明此前步骤已处理过
                # 一部分 block 失败。
                req_num_computed_tokens = (
                    request.num_computed_tokens
                    if req_id in self.failed_recving_kv_req_ids
                    else len(req_block_ids) * self.block_size
                )
            else:
                # 同步加载：num_computed_tokens 包含本轮新 token。
                req_num_computed_tokens = request.num_cached_tokens

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # 该无效 block 与前序请求共享，且已被标记重算；
                    # 因此本请求重调度时仍可视其为“可由共享结果覆盖”。
                    # 目前仅同步加载支持 block 共享；异步加载暂不支持。
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # 该请求已经标记过一个无效 block 并更新过 num_computed_tokens。
                    continue

                marked_invalid_block = True
                # 在第一个失败 block 处截断 computed token 计数。
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens
                request.num_external_computed_tokens -= num_affected_tokens
                # 收集该无效 block 及其后续依赖 block，供驱逐使用。
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # 本请求的无效 block 都是共享 block，会由前序请求重算。
                    # 这里回退为“仅把已缓存 token 视作已计算”。
                    # 目前仅同步加载支持 block 共享；异步加载暂不支持。
                    total_affected_tokens += (
                        request.num_computed_tokens - request.num_cached_tokens
                    )
                    request.num_computed_tokens = request.num_cached_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        """处理受无效 KV block 影响的请求。

        返回：
            update_from_output 主循环中应跳过的受影响请求 ID 集合。
        """
        should_fail = not self.recompute_kv_load_failures

        # 处理异步 KV 加载（尚未缓存，因此 evict_blocks=False）。
        async_load_reqs = (
            req
            for req in self.waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs, invalid_block_ids, evict_blocks=False
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # 处理同步 KV 加载（可能已缓存，需要收集待驱逐 block）。
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # 仅在非 recompute 策略下驱逐无效及下游依赖 block。
        # recompute 策略下这些 block 会重算，并可被共享请求复用。
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # 标记异步加载失败请求：待加载完成后重试。
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # 返回同步路径受影响请求，供 update_from_output 跳过。
        return sync_failed_req_ids
