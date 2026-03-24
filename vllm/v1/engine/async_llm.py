# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import socket
import time
import warnings
from collections.abc import AsyncGenerator, Iterable, Mapping
from copy import copy
from typing import Any

import torch

import vllm.envs as envs
from vllm import TokensPrompt
from vllm.config import VllmConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.entrypoints.serve.elastic_ep.middleware import set_scaling_elastic_ep
from vllm.inputs import ProcessorInputs, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import STREAM_FINISHED, PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.tracing import init_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.usage.usage_lib import UsageContext
from vllm.utils.async_utils import cancel_task_threadsafe
from vllm.utils.collection_utils import as_list
from vllm.v1.engine import EngineCoreRequest, PauseMode
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import (
    StatLoggerFactory,
    StatLoggerManager,
    load_stat_logger_plugin_factories,
)
from vllm.v1.metrics.prometheus import shutdown_prometheus
from vllm.v1.metrics.stats import IterationStats

logger = init_logger(__name__)


class InputStreamError(Exception):
    """包装输入流生成器抛出的异常。

    这样 generate() 可以把用户输入流里的原始异常继续往上抛，
    而不是统一包成 EngineGenerateError。
    """

    def __init__(self, cause: Exception):
        self.cause = cause
        super().__init__(str(cause))


class AsyncLLM(EngineClient):
    """vLLM V1 运行时的异步前端封装。

    可以把它理解成 API Server 和 EngineCore 之间的桥接层：
    - 向上接收 chat/completion/embedding 等请求
    - 在本进程里做输入预处理和输出整理
    - 向下通过 EngineCoreClient 把请求发给真正的调度/执行核心
    - 通过后台 output_handler 把 EngineCore 的输出回流给各请求队列
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        """初始化 AsyncLLM。

        这个类本身不负责真正的调度或模型执行，它主要负责 4 件事：
        1. 维护 renderer / tokenizer / io_processor 等前端组件
        2. 用 InputProcessor 把上层输入转成 EngineCoreRequest
        3. 用 OutputProcessor 把 EngineCoreOutput 转成 RequestOutput
        4. 通过 EngineCoreClient 与后端核心进程通信
        """
        # 确保自定义 transformer config 可以跨进程序列化。
        maybe_register_config_serialize_by_value()

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.observability_config = vllm_config.observability_config

        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        if tracing_endpoint is not None:
            init_tracer("vllm.llm_engine", tracing_endpoint)

        self.log_requests = log_requests

        custom_stat_loggers = list(stat_loggers or [])
        custom_stat_loggers.extend(load_stat_logger_plugin_factories())

        has_custom_loggers = bool(custom_stat_loggers)
        self.log_stats = log_stats or has_custom_loggers
        if not log_stats and has_custom_loggers:
            logger.info(
                "AsyncLLM created with log_stats=False, "
                "but custom stat loggers were found; "
                "enabling logging without default stat loggers."
            )

        # renderer 负责 prompt 渲染、tokenizer 获取等前端输入输出相关能力。
        self.renderer = renderer = renderer_from_config(self.vllm_config)
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.renderer,
            self.model_config.io_processor_plugin,
        )

        # 把上层 prompt / sampling params 规范化成 EngineCoreRequest。
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # 把后端核心进程返回的输出转成 API 层可消费的 RequestOutput。
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # 真正的调度与执行在 EngineCore 中完成；这里拿到的是它的异步 client。
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

        # 统计与指标日志管理器。
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=custom_stat_loggers,
                enable_default_loggers=log_stats,
                client_count=client_count,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        self._client_count = client_count

        self.output_handler: asyncio.Task | None = None
        try:
            # 如果当前已经在事件循环中，就提前把后台输出循环拉起来。
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

        if (
            vllm_config.profiler_config.profiler == "torch"
            and not vllm_config.profiler_config.ignore_frontend
        ):
            profiler_dir = vllm_config.profiler_config.torch_profiler_dir
            logger.info(
                "Torch profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501
                profiler_dir,
            )
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_llm"
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=vllm_config.profiler_config.torch_profiler_with_stack,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    profiler_dir,
                    worker_name=worker_name,
                    use_gzip=vllm_config.profiler_config.torch_profiler_use_gzip,
                ),
            )
        else:
            self.profiler = None

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncLLM":
        # 根据配置选择合适的 Executor，再构建 AsyncLLM。
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> "AsyncLLM":
        """根据 EngineArgs 构建 AsyncLLM。"""

        # 先把命令行 / 外部参数转成内部统一配置。
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # 再创建前端异步 engine 封装。
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """关闭 AsyncLLM，并清理后台进程、IPC 与本地任务。"""

        shutdown_prometheus()

        if renderer := getattr(self, "renderer", None):
            renderer.shutdown()

        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown()

        handler = getattr(self, "output_handler", None)
        if handler is not None:
            cancel_task_threadsafe(handler)

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        if not hasattr(self, "_supported_tasks"):
            # 支持的任务类型由后端 engine 决定，这里只查询一次并缓存。
            self._supported_tasks = await self.engine_core.get_supported_tasks_async()

        return self._supported_tasks

    async def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest
        | PromptType
        | ProcessorInputs
        | AsyncGenerator[StreamingInput, None],
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        prompt_text: str | None = None,
        reasoning_ended: bool | None = None,
    ) -> RequestOutputCollector:
        """把一个新请求接入 AsyncLLM，并返回该请求对应的输出收集器。

        这是 generate()/encode() 共同依赖的核心入口。主流程是：
        1. 校验请求与参数
        2. 把上层输入转成 EngineCoreRequest
        3. 给请求分配内部 request_id
        4. 在 OutputProcessor 中登记输出队列
        5. 把请求发给 EngineCore
        """

        if self.errored:
            raise EngineDeadError()

        is_pooling = isinstance(params, PoolingParams)

        if (
            self.vllm_config.cache_config.kv_sharing_fast_prefill
            and not is_pooling
            and params.prompt_logprobs
        ):
            raise ValueError(
                "--kv-sharing-fast-prefill produces incorrect logprobs for "
                "prompt tokens, please disable it when the requests need "
                "prompt logprobs"
            )

        if isinstance(prompt, AsyncGenerator):
            if reasoning_ended is not None:
                raise NotImplementedError

            # 输入本身是异步流时，走“边收输入边送后端”的特殊路径。
            return await self._add_streaming_input_request(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
                data_parallel_rank,
            )

        # 普通请求会在这里被规范化成内部的 EngineCoreRequest。
        if isinstance(prompt, EngineCoreRequest):
            logger.warning_once(
                "Passing EngineCoreRequest to AsyncLLM.generate() and .add_requests() "
                "is deprecated and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "AsyncLLM.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                supported_tasks=await self.get_supported_tasks(),
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
            )
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

        if reasoning_ended is not None:
            request.reasoning_ended = reasoning_ended

        # 对外部 request_id 做内部标准化，保证整个运行时里是唯一、可追踪的。
        self.input_processor.assign_request_id(request)

        # 后台输出循环延迟到第一次收请求时再启动，这样 AsyncLLM 可以在没有
        # event loop 的上下文里被安全创建。
        self._run_output_handler()

        # 每个请求都有自己独立的输出收集队列，generate()/encode() 就从这里取结果。
        queue = RequestOutputCollector(params.output_kind, request.request_id)

        # process_inputs() 可能会克隆或修正参数，这里以后续版本为准。
        params = request.params

        if is_pooling or params.n == 1:
            await self._add_request(request, prompt_text, None, 0, queue)
            return queue

        parent_params = params
        assert isinstance(parent_params, SamplingParams)

        # n > 1 时，一个上层请求会拆成多个子请求并共享同一个父级 collector。
        parent_request = ParentRequest(request)
        for idx in range(parent_params.n):
            request_id, child_params = parent_request.get_child_info(idx)
            child_request = request if idx == parent_params.n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params
            await self._add_request(
                child_request, prompt_text, parent_request, idx, queue
            )
        return queue

    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ):
        # 先在当前进程登记请求，建立 request_id -> queue 的映射关系。
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # 再把真正的 EngineCoreRequest 发给后端核心进程。
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)

    async def _add_streaming_input_request(
        self,
        request_id: str,
        input_stream: AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> RequestOutputCollector:
        """处理“输入本身就是异步流”的请求。

        这条路径会启动一个本地协程，持续消费 input_stream，把每个输入片段
        转成可恢复的请求增量并送往 EngineCore。
        """
        self._validate_streaming_input_sampling_params(sampling_params)

        inputs = dict(
            supported_tasks=await self.get_supported_tasks(),
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )

        if not sampling_params.skip_clone:
            sampling_params = sampling_params.clone()
            sampling_params.skip_clone = True

        # 先构造一个“最终空请求”做参数校验；输入流结束时，还会用它作为
        # “输入已经全部送完”的结束信号发给后端。
        final_req = self.input_processor.process_inputs(
            request_id=request_id,
            prompt=TokensPrompt(prompt_token_ids=[0]),
            params=sampling_params,
            **inputs,  # type: ignore[arg-type]
        )
        self.input_processor.assign_request_id(final_req)
        internal_req_id = final_req.request_id

        queue = RequestOutputCollector(sampling_params.output_kind, internal_req_id)

        async def handle_inputs():
            cancelled = False
            try:
                async for input_chunk in input_stream:
                    sp = input_chunk.sampling_params
                    if sp:
                        self._validate_streaming_input_sampling_params(sp)
                    else:
                        sp = sampling_params
                    # TODO(nick): 避免对复用的 sampling_params 重复校验。
                    req = self.input_processor.process_inputs(
                        request_id=internal_req_id,
                        prompt=input_chunk.prompt,
                        params=sp,
                        resumable=True,
                        **inputs,  # type: ignore[arg-type]
                    )
                    req.external_req_id = request_id
                    if req.prompt_embeds is not None:
                        raise ValueError(
                            "prompt_embeds not supported for streaming inputs"
                        )
                    prompt_text, _, _ = extract_prompt_components(
                        self.model_config, input_chunk.prompt
                    )
                    await self._add_request(req, prompt_text, None, 0, queue)
            except (asyncio.CancelledError, GeneratorExit):
                cancelled = True
            except Exception as error:
                # 包一层 InputStreamError，让 generate() 能保留用户输入流的原始异常。
                queue.put(InputStreamError(error))
            finally:
                queue._input_stream_task = None
                if not cancelled:
                    # 发送一个空的 final request，告诉后端“输入流已经结束”。
                    # 如果会话已经取消，则不再发送，避免多余状态推进。
                    await self._add_request(final_req, None, None, 0, queue)

        # 输入流开始前，也要确保后台输出循环已经启动。
        self._run_output_handler()

        queue._input_stream_task = asyncio.create_task(handle_inputs())
        return queue

    @staticmethod
    def _validate_streaming_input_sampling_params(
        params: SamplingParams | PoolingParams,
    ):
        """校验输入流模式下允许使用的参数范围。"""
        if (
            not isinstance(params, SamplingParams)
            or params.n > 1
            or params.output_kind == RequestOutputKind.FINAL_ONLY
            or params.stop
        ):
            raise ValueError(
                "Input streaming not currently supported "
                "for pooling models, n > 1, request_kind = FINAL_ONLY "
                "or with stop strings."
            )

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    async def generate(
        self,
        prompt: EngineCoreRequest
        | PromptType
        | ProcessorInputs
        | AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """文本生成主入口。

        API Server 最终会走到这里。执行顺序可以概括成：
        1. 调用 add_request() 把请求送入 AsyncLLM
        2. add_request() 再把请求登记到 OutputProcessor，并发给 EngineCore
        3. 后台 _run_output_handler() 持续从 EngineCore 拉取输出
        4. 当前协程从该请求对应的队列中取出 RequestOutput 并 yield 给上层
        """

        q: RequestOutputCollector | None = None
        try:
            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                prompt_text=prompt_text,
                reasoning_ended=reasoning_ended,
            )

            # output_handler 负责往队列里塞数据；当前协程负责从队列里取数据返回给调用方。
            finished = False
            while not finished:
                # 优先无 await 地取数据，能少一次事件循环切换，对高并发路径更友好。
                out = q.get_nowait() or await q.get()

                # 请求结束后的清理由 OutputProcessor 和 EngineCore 各自负责。
                assert isinstance(out, RequestOutput)
                finished = out.finished
                if out is not STREAM_FINISHED:
                    yield out

        # 客户端断开连接、协程被取消或生成器被回收时，需要主动中止后端请求。
        except (asyncio.CancelledError, GeneratorExit):
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # 后端已经挂掉时，不再重复 abort。
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # 参数校验错误直接透传。
        except ValueError as e:
            if self.log_requests:
                logger.info("Request %s failed (bad request): %s.", request_id, e)
            raise

        # 输入流里的原始异常直接继续向上抛。
        except InputStreamError as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed (input error): %s.", request_id, e)
            raise e.cause from e

        # 其余异常统一包装成 EngineGenerateError。
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                try:
                    s = f"{e.__class__.__name__}: {e}"
                except Exception as e2:
                    s = (
                        f"{e.__class__.__name__}: "
                        "error during printing an exception of class"
                        + e2.__class__.__name__
                    )
                logger.info("Request %s failed due to %s.", request_id, s)
            raise EngineGenerateError() from e
        finally:
            if q is not None:
                q.close()

    def _run_output_handler(self):
        """启动后台输出循环，把 EngineCore 的输出分发回各请求队列。"""

        if self.output_handler is not None:
            return

        # 下面把成员引用拆出来，是为了避免 task 闭包反向持有 self，
        # 导致 AsyncLLM 无法及时被 GC。
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        # logger_manager 用可变 list 包一层，这样 elastic scaling 后可以替换，
        # 同时又不必让闭包直接引用 self。
        self._logger_ref = [self.logger_manager]
        logger_ref = self._logger_ref
        renderer = self.renderer
        chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE

        async def output_handler():
            try:
                while True:
                    # 1. 从后端核心进程拉取这一轮完成的输出。
                    outputs = await engine_core.get_output_async()
                    num_outputs = len(outputs.outputs)

                    iteration_stats = (
                        IterationStats() if (log_stats and num_outputs) else None
                    )

                    # 把大批量输出切成小块处理，避免单次处理占用事件循环太久。
                    engine_core_outputs = outputs.outputs
                    for start in range(0, num_outputs, chunk_size):
                        end = start + chunk_size
                        outputs_slice = engine_core_outputs[start:end]
                        # 2. 把 EngineCoreOutput 转成 RequestOutput，并分发到对应请求队列。
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats
                        )
                        # RequestOutput 在 process_outputs() 内部已经被推入对应 queue。
                        assert not processed_outputs.request_outputs

                        # 在分块之间让出控制权，避免长时间阻塞其他 asyncio 任务。
                        if end < num_outputs:
                            await asyncio.sleep(0)

                        # 3. 某些请求可能因为 stop string 命中而需要回头通知后端 abort。
                        if processed_outputs.reqs_to_abort:
                            await engine_core.abort_requests_async(
                                processed_outputs.reqs_to_abort
                            )

                    output_processor.update_scheduler_stats(outputs.scheduler_stats)

                    # 4. 记录调度/迭代统计信息。
                    # TODO(rob): Prometheus 开销变大后，再考虑搬到后台线程。
                    if logger_ref[0]:
                        logger_ref[0].record(
                            engine_idx=outputs.engine_index,
                            scheduler_stats=outputs.scheduler_stats,
                            iteration_stats=iteration_stats,
                            mm_cache_stats=renderer.stat_mm_cache(),
                        )
            except Exception as e:
                logger.exception("AsyncLLM output_handler failed.")
                # 让所有等待该请求输出的调用方都能感知到这个后台错误。
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(output_handler())

    async def abort(
        self, request_id: str | Iterable[str], internal: bool = False
    ) -> None:
        """同时在 OutputProcessor 和 EngineCore 中中止请求。"""

        request_ids = (
            (request_id,) if isinstance(request_id, str) else as_list(request_id)
        )
        all_request_ids = self.output_processor.abort_requests(request_ids, internal)
        await self.engine_core.abort_requests_async(all_request_ids)

        if self.log_requests:
            logger.info("Aborted request(s) %s.", ",".join(request_ids))

    async def pause_generation(
        self,
        *,
        mode: PauseMode = "abort",
        wait_for_inflight_requests: bool | None = None,
        clear_cache: bool = True,
    ) -> None:
        """暂停调度，常用于更新权重或做运行时切换。

        真正的暂停/排空/保留队列逻辑都在 EngineCore 里完成。暂停期间，
        新的 generate/encode 请求不会继续被调度执行。
        """
        if wait_for_inflight_requests:
            warnings.warn(
                "The `wait_for_inflight_requests` parameter in "
                "`AsyncLLM.pause_generation()` is deprecated. "
                "Please use `mode` argument instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "wait"
        await self.engine_core.pause_scheduler_async(mode=mode, clear_cache=clear_cache)
        # 稍微 sleep 一下，让“已经在路上的最后一批输出”有机会先经过 output_handler
        # 回到调用方，避免从调用者视角看起来事件顺序很奇怪。
        await asyncio.sleep(0.02)

    async def resume_generation(self) -> None:
        """恢复暂停后的调度执行。"""
        await self.engine_core.resume_scheduler_async()

    async def is_paused(self) -> bool:
        """返回当前调度器是否处于暂停状态。"""
        return await self.engine_core.is_scheduler_paused_async()

    async def encode(
        self,
        prompt: PromptType | ProcessorInputs,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
        reasoning_ended: bool | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """pooling/embedding 主入口。

        它和 generate() 的整体结构基本一致，只是输出类型换成了
        PoolingRequestOutput。
        """

        q: RequestOutputCollector | None = None
        try:
            q = await self.add_request(
                request_id,
                prompt,
                pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                reasoning_ended=reasoning_ended,
            )

            # output_handler 写队列，这里读队列并把 pooling 结果返回给调用方。
            finished = False
            while not finished:
                # 同样优先走无 await 的快路径。
                out = q.get_nowait() or await q.get()
                assert isinstance(out, PoolingRequestOutput)
                # 请求结束后的状态清理由 OutputProcessor / EngineCore 各自处理。
                finished = out.finished
                yield out

        # 如果调用方取消，就中止后端请求。
        except asyncio.CancelledError:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # 后端挂掉时不再重复 abort。
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # 参数校验错误直接透传。
        except ValueError:
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        # 其余异常统一包装成 EngineGenerateError。
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e
        finally:
            if q is not None:
                q.close()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    async def is_tracing_enabled(self) -> bool:
        return self.observability_config.otlp_traces_endpoint is not None

    async def do_log_stats(self) -> None:
        if self.logger_manager:
            self.logger_manager.log()

    async def check_health(self) -> None:
        logger.debug("Called check_health.")
        if self.errored:
            raise self.dead_error

    async def start_profile(self, profile_prefix: str | None = None) -> None:
        coros = [self.engine_core.profile_async(True, profile_prefix)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.start))
        await asyncio.gather(*coros)

    async def stop_profile(self) -> None:
        coros = [self.engine_core.profile_async(False)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.stop))
        await asyncio.gather(*coros)

    async def reset_mm_cache(self) -> None:
        self.renderer.clear_mm_cache()
        await self.engine_core.reset_mm_cache_async()

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return await self.engine_core.reset_prefix_cache_async(
            reset_running_requests, reset_connector
        )

    async def reset_encoder_cache(self) -> None:
        await self.engine_core.reset_encoder_cache_async()

    async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        await self.engine_core.sleep_async(level, mode)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        await self.engine_core.wake_up_async(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    async def is_sleeping(self) -> bool:
        return await self.engine_core.is_sleeping_async()

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """向后端 engine 动态加载一个新的 LoRA 适配器。"""
        return await self.engine_core.add_lora_async(lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        """卸载一个已经加载的 LoRA 适配器。"""
        return await self.engine_core.remove_lora_async(lora_id)

    async def list_loras(self) -> set[int]:
        """列出当前已注册的 LoRA 适配器。"""
        return await self.engine_core.list_loras_async()

    async def pin_lora(self, lora_id: int) -> bool:
        """把指定 LoRA 标记为不可驱逐。"""
        return await self.engine_core.pin_lora_async(lora_id)

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """向所有相关 engine worker 发起一次集体 RPC。"""
        return await self.engine_core.collective_rpc_async(
            method, timeout, args, kwargs
        )

    async def wait_for_requests_to_drain(self, drain_timeout: int = 300):
        """等待当前所有请求自然排空。"""
        start_time = time.time()
        while time.time() - start_time < drain_timeout:
            if not self.engine_core.dp_engines_running():
                logger.info("Engines are idle, requests have been drained")
                return

            logger.info("Engines are still running, waiting for requests to drain...")
            await asyncio.sleep(1)  # 每秒轮询一次空闲状态

        raise TimeoutError(
            f"Timeout reached after {drain_timeout} seconds "
            "waiting for requests to drain."
        )

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ):
        """弹性调整 data parallel 大小。

        本质上是增加或减少后端 engine core 的数量。必要时会先等现有请求排空，
        再更新统计组件和 EngineCore 的并行规模。
        """
        old_data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        if old_data_parallel_size == new_data_parallel_size:
            logger.info(
                "Data parallel size is already %s, skipping scale",
                new_data_parallel_size,
            )
            return

        if envs.VLLM_ELASTIC_EP_DRAIN_REQUESTS:
            logger.info(
                "VLLM_ELASTIC_EP_DRAIN_REQUESTS is set, "
                "waiting for requests to drain before scaling"
            )
            await self.wait_for_requests_to_drain(drain_timeout)

        # 扩容时需要重建 stat logger，让新 engine rank 也能被纳入统计。
        if new_data_parallel_size > old_data_parallel_size and self.log_stats:
            # TODO(rob): fix this after talking with Ray team.
            # This resets all the prometheus metrics since we
            # unregister during initialization. Need to understand
            # the intended behavior here better.
            self.logger_manager = StatLoggerManager(
                vllm_config=self.vllm_config,
                engine_idxs=list(range(new_data_parallel_size)),
                custom_stat_loggers=None,
            )
            # 更新可变引用，让 output_handler 能感知到新的 logger_manager。
            if hasattr(self, "_logger_ref"):
                self._logger_ref[0] = self.logger_manager
            self.logger_manager.log_engine_initialized()

        set_scaling_elastic_ep(True)
        try:
            await self.engine_core.scale_elastic_ep(new_data_parallel_size)
            self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        finally:
            set_scaling_elastic_ep(False)

    @property
    def is_running(self) -> bool:
        # output_handler 在首次收请求前可能还是 None，此时也认为是“可运行”状态。
        return self.output_handler is None or not self.output_handler.done()

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return self.engine_core.resources.engine_dead or not self.is_running

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()

    async def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest
    ) -> None:
        """为 RL 训练场景初始化权重传输引擎。"""
        from vllm.distributed.weight_transfer.base import (
            WeightTransferInitRequest,
        )

        if isinstance(request, WeightTransferInitRequest):
            init_info_dict = request.init_info
        else:
            raise TypeError(f"Expected WeightTransferInitRequest, got {type(request)}")

        await self.collective_rpc(
            "init_weight_transfer_engine", kwargs={"init_info": init_info_dict}
        )

    async def update_weights(self, request: WeightTransferUpdateRequest) -> None:
        """为 RL 训练场景执行一次批量权重更新。"""

        if isinstance(request, WeightTransferUpdateRequest):
            update_info_dict = request.update_info
        else:
            raise TypeError(
                f"Expected WeightTransferUpdateRequest, got {type(request)}"
            )

        await self.collective_rpc(
            "update_weights", kwargs={"update_info": update_info_dict}
        )
