# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
import traceback
import weakref
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property, partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Lock as LockType
from threading import Thread
from typing import Any, cast

import cloudpickle
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_dp_group,
    get_ep_group,
    get_inner_dp_world_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    model_parallel_is_initialized,
)
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.tracing import instrument, maybe_init_worker_tracer
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_loopback_ip,
    get_open_port,
)
from vllm.utils.system_utils import (
    _maybe_force_spawn,
    decorate_logs,
    get_mp_context,
    set_process_title,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class FutureWrapper(Future):
    """对多进程 RPC 结果做延迟提取的 Future 包装。

    `MultiprocExecutor.collective_rpc(non_block=True)` 不会立刻去各个 MQ
    阻塞收结果，而是把“如何取结果”的闭包塞进队列里，等调用方真正
    `.result()` 时再按顺序回收。这个类就是那层薄包装。
    """

    def __init__(
        self,
        futures_queue: deque[tuple["FutureWrapper", Callable]],
        aggregate: Callable = lambda x: x,
    ):
        self.futures_queue = futures_queue
        self.aggregate = aggregate
        super().__init__()

    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")
        # 先把排在自己前面的 future 都取完，保证响应消费顺序与发送顺序一致。
        while not self.done():
            future, get_response = self.futures_queue.pop()
            future.wait_for_response(get_response)
        return super().result()

    def wait_for_response(self, get_response: Callable):
        try:
            response = self.aggregate(get_response())
            with suppress(InvalidStateError):
                self.set_result(response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)


class MultiprocExecutor(Executor):
    """基于本地多进程的 executor 实现。

    你可以把它理解成“worker 进程集群的父进程控制器”：

    1. 父进程负责拉起多个 worker 子进程
    2. 通过共享内存消息队列把 RPC / SchedulerOutput 广播给所有 worker
    3. 从一个或多个 worker 回收结果
    4. 监控 worker 存活状态，出问题时触发整体关闭
    """

    supports_pp: bool = True

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):
        self.monitor_workers = monitor_workers
        super().__init__(vllm_config)

    def _init_executor(self) -> None:
        # 在 executor 对象被回收时兜底调用 shutdown，确保子进程不会泄漏。
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: FailureCallback | None = None

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )

        # 在拉起子进程前统一配置多进程环境变量，例如线程数、spawn 策略等。
        set_multiprocessing_worker_envs()

        # 本地节点内的分布式初始化使用 loopback 地址即可。
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port()
        )
        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle: Handle | None = None
        # 运行主线：
        # 1. leader executor 创建广播队列，用于把 RPC / SchedulerOutput 发给 worker
        # 2. 再创建本地 worker 进程
        # 3. 等 worker 初始化完成并上报自己的 response MQ handle
        # 4. 父进程收集所有 response MQ，之后即可开始 collective_rpc
        if self.parallel_config.node_rank_within_dp == 0:
            # 每个 DP 内部只有 leader node 负责持有广播 MQ。
            # 也就是说，每个 DP 组都会有自己的一个 MultiprocExecutor leader。
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            mq_connect_ip = get_ip()
            logger.info(
                "DP group leader: node_rank=%d, node_rank_within_dp=%d, "
                "master_addr=%s, mq_connect_ip=%s (local), "
                "world_size=%d, local_world_size=%d",
                self.parallel_config.node_rank,
                self.parallel_config.node_rank_within_dp,
                self.parallel_config.master_addr,
                mq_connect_ip,
                self.world_size,
                self.local_world_size,
            )
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=mq_connect_ip,
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        # 创建本机上的 worker 子进程。
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            global_start_rank = (
                self.local_world_size * self.parallel_config.node_rank_within_dp
            )
            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                is_driver_worker = self._is_driver_worker(global_rank)
                unready_workers.append(
                    WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=local_rank,
                        rank=global_rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                        shared_worker_lock=shared_worker_lock,
                        is_driver_worker=is_driver_worker,
                    )
                )

            # 必须先把所有 worker 都创建出来，再统一等待 READY。
            # 否则 worker.init_device() 中的设备同步可能导致死锁。

            # 等待所有本地 worker 完成初始化。
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # 启动后台线程监控 worker 存活状态。
            if self.monitor_workers:
                self.start_worker_monitor()

            self.response_mqs = []
            # 只有 leader node 需要收集跨节点 response MQ。
            if self.parallel_config.node_rank_within_dp == 0:
                for rank in range(self.world_size):
                    if rank < self.local_world_size:
                        local_message_queue = self.workers[rank].worker_response_mq
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)
                    else:
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[
                            rank
                        ]
                        assert remote_message_queue is not None
                        self.response_mqs.append(remote_message_queue)

            # 等待各 MQ 完成握手。
            # 注意这里的顺序必须与 WorkerProc 中保持一致，否则会死锁。

            # 先等输入广播队列 ready。
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            # 再等所有 response 队列 ready。
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()

            # non_block RPC 的延迟回收队列。
            self.futures_queue = deque[tuple[FutureWrapper, Callable]]()

            self._post_init_executor()

            success = True
        finally:
            if not success:
                # 初始化失败时，尽可能把已拉起的子进程清理干净。
                # 先关闭 death_writer，通知子进程退出。
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        # output_rank 表示“谁负责把最终 ModelRunnerOutput 回给父进程”。
        self.output_rank = self._get_output_rank()

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        self.world_size = self.parallel_config.world_size
        assert self.world_size % self.parallel_config.nnodes_within_dp == 0, (
            f"global world_size ({self.parallel_config.world_size}) must be "
            f"divisible by nnodes_within_dp "
            f"({self.parallel_config.nnodes_within_dp}). "
        )
        self.local_world_size = self.parallel_config.local_world_size
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        return tp_size, pp_size, pcp_size

    def _post_init_executor(self) -> None:
        pass

    def _is_driver_worker(self, rank: int) -> bool:
        return rank % self.parallel_config.tensor_parallel_size == 0

    def start_worker_monitor(self, inline=False) -> None:
        workers = self.workers
        self_ref = weakref.ref(self)

        # 监控 worker 进程存活状态。
        # 任意一个 worker 异常退出，都视为 executor 进入失败状态：
        # 记录日志、关闭 executor，并通过 failure_callback 通知上层 engine。
        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, "shutting_down", False):
                return
            _self.is_failed = True
            proc_name = next(h.proc.name for h in workers if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly, shutting down executor.", proc_name
            )
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        if not inline:
            Thread(
                target=monitor_workers, daemon=True, name="MultiprocWorkerMonitor"
            ).start()
            return

        monitor_workers()

    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
        )

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
        )

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch", unique_reply_rank=self.output_rank)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # 优化：draft token 只需从单个输出 worker（output_rank）回收。
        return self.collective_rpc(
            "take_draft_token_ids", unique_reply_rank=self.output_rank
        )

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator | None = None,
    ) -> Any:
        """在所有 worker 上广播一次 RPC，并回收结果。

        这是多进程 executor 的核心控制面接口。整体流程是：

        1. 父进程把 `(method, args, kwargs, output_rank)` 广播到所有 worker
        2. 每个 worker 在自己的 busy loop 中执行该方法
        3. 需要回包的 worker 把结果写入 response MQ
        4. 父进程从一个或多个 response MQ 中取回结果

        当设置了 `unique_reply_rank` 时，只等待指定 rank 的回复；
        当设置了 `kv_output_aggregator` 时，会对多个 worker 的结果进一步聚合。
        """
        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called on follower node"
        )
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        if kv_output_aggregator is not None:
            # 存在聚合器时，需要先收齐所有相关 worker 的回复，再做聚合。
            output_rank = None
            aggregate: Callable[[Any], Any] = partial(
                kv_output_aggregator.aggregate, output_rank=unique_reply_rank or 0
            )
        else:
            output_rank = unique_reply_rank
            aggregate = lambda x: x

        if isinstance(method, str):
            send_method = method
        else:
            # 可调用对象会被 cloudpickle 序列化后发给 worker 执行。
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))

        response_mqs: Sequence[MessageQueue] = self.response_mqs
        if output_rank is not None:
            # 只期待某一个 rank 回包时，只监听对应的 response MQ。
            response_mqs = (response_mqs[output_rank],)

        shutdown_event = self.shutdown_event

        def get_response():
            responses = []
            for mq in response_mqs:
                dequeue_timeout = (
                    None if deadline is None else (deadline - time.monotonic())
                )
                try:
                    status, result = mq.dequeue(
                        timeout=dequeue_timeout, cancel=shutdown_event
                    )
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e
                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause"
                    )
                responses.append(result)
            return responses[0] if output_rank is not None else responses

        if non_block:
            # non_block 模式下不立刻收包，而是把“如何收包”包装成 Future。
            future = FutureWrapper(self.futures_queue, aggregate=aggregate)
            self.futures_queue.appendleft((future, get_response))
            return future

        # 阻塞模式下，先把前面挂起的 non_block future 统一回收掉，
        # 保证响应消费顺序稳定。
        while self.futures_queue:
            future, get_fut_response = self.futures_queue.pop()
            future.wait_for_response(get_fut_response)

        return aggregate(get_response())

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """确保所有 worker 子进程最终退出。

        假设调用方已经向 worker 发出了退出信号。这里会：
        1. 先等待一小段时间，给正常清理留机会
        2. 若仍未退出，则发送 SIGTERM
        3. 若还未退出，则发送 SIGKILL
        """

        def wait_for_termination(procs, timeout):
            if not time:
                # 解释器退出晚期，模块级 `time` 可能已经被置空。
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        active_procs = lambda: [proc for proc in worker_procs if proc.is_alive()]
        # 先给子进程留一点时间自行收尾。
        if wait_for_termination(active_procs(), 4):
            return

        # 仍未退出则发送 SIGTERM。
        for p in active_procs():
            p.terminate()
        if not wait_for_termination(active_procs(), 4):
            # 还不退出则强制 SIGKILL。
            for p in active_procs():
                p.kill()

    def shutdown(self):
        """有序关闭 executor 及其所有 worker。"""
        if not getattr(self, "shutting_down", False):
            self.shutting_down = True

            # 先确保所有子进程退出。
            if workers := getattr(self, "workers", None):
                for w in workers:
                    # 关闭 death_writer，让子进程侧的 death monitor 感知父进程退出。
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                    w.worker_response_mq = None
                self._ensure_worker_termination([w.proc for w in workers])

            self.shutdown_event.set()

        self.rpc_broadcast_mq = None

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return

    @cached_property
    def max_concurrent_batches(self) -> int:
        # PP 场景下，通常需要与 PP stage 数相当的并发 batch 才能把流水线填满。
        pp_size = self.parallel_config.pipeline_parallel_size
        return 2 if pp_size <= 1 and self.scheduler_config.async_scheduling else pp_size

    def _get_output_rank(self) -> int:
        # 最终只从“最后一个 PP stage 的第一个 TP worker”收集 ModelRunnerOutput。
        # 这是因为：
        # 1. 最终输出只会在最后一个 PP stage 产生
        # 2. 同一个 TP 组里通常只需由 rank 0 代表回包
        #
        # 例子：
        # 假设 TP=8, PP=4，则 world_size=32
        # 0-7   -> PP rank 0
        # 8-15  -> PP rank 1
        # 16-23 -> PP rank 2
        # 24-31 -> PP rank 3（最后一个 stage）
        # 因此 output rank = 24
        return (
            self.world_size
            - self.parallel_config.tensor_parallel_size
            * self.parallel_config.prefill_context_parallel_size
        )


@dataclass
class UnreadyWorkerProcHandle:
    """worker 进程在进入 READY 之前的句柄。"""

    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Connection | None = None


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    # 单机模式下，该 worker 会把执行结果直接写入这个 MQ。
    worker_response_mq: MessageQueue | None
    # 仅在 driver node 上非空。
    # 第 i 个远端 worker 会把结果写到 `peer_worker_response_mqs[i]` 对应的 MQ。
    peer_worker_response_mqs: list[MessageQueue | None]
    death_writer: Connection | None = None

    @classmethod
    def from_unready_handle(
        cls,
        unready_handle: UnreadyWorkerProcHandle,
        worker_response_mq: MessageQueue | None,
        peer_worker_response_mqs: list[MessageQueue | None],
    ) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
            death_writer=unready_handle.death_writer,
        )


class WorkerProc:
    """在独立子进程中运行一个 Worker 的包装器。"""

    READY_STR = "READY"
    rpc_broadcast_mq: MessageQueue | None
    worker_response_mq: MessageQueue | None

    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        if vllm_config.parallel_config.nnodes_within_dp == 1:
            # 单机模式：
            # 1. 从父进程导出的 handle 恢复输入广播队列，用于接收 RPC/SchedulerOutput
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.worker.rank
            )

            # 2. 再创建一个本地 response MQ，把模型输出回传给父进程
            self.worker_response_mq = MessageQueue(1, 1)
            self.peer_response_handles = []
        else:
            # 多节点 DP 模式：
            # 1. 通过 inner DP world group 创建跨节点广播接收端
            self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(
                external_writer_handle=input_shm_handle,
                # 这里已有来自 executor 进程的 external_writer_handle，
                # 因此创建时先不阻塞等待握手，后续在 wait_until_ready() 再统一触发。
                blocking=False,
            )
            # 2. 创建跨节点 response MQ，把模型输出发回 driver worker；
            # 同时暴露 peer_response_handles，让 driver 侧知道所有 rank 的句柄
            self.worker_response_mq, self.peer_response_handles = (
                get_inner_dp_world_group().create_single_reader_mq_broadcasters(
                    reader_rank_in_group=0
                )
            )

    @instrument(span_name="Worker init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
        # TODO: 未来可以把 `init_worker` 提升到 executor 层做一次 collective rpc。
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
            "shared_worker_lock": shared_worker_lock,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling
        if self.use_async_scheduling:
            # 异步调度模式下，GPU 侧输出可能需要额外线程异步拷回 CPU 并入队。
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy",
            )
            self.async_output_copy_thread.start()

        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )

        # 子进程初始化主线：
        # 1. 建立输入/输出消息队列
        # 2. 初始化设备与分布式环境
        # 3. 加载模型
        self._init_message_queues(input_shm_handle, vllm_config)
        is_eep_new_worker = envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH
        if not is_eep_new_worker:
            self.worker.init_device()
            # 并行组初始化完成后，进程名与日志前缀可以带上 DP/TP/PP 等信息。
            self.setup_proc_title_and_log_prefix(
                enable_ep=vllm_config.parallel_config.enable_expert_parallel
            )
            self.worker.load_model()

        # 启用环境变量缓存。到这一步为止，通常不会再动态覆写 env。
        enable_envs_cache()

    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # 用于接收 SchedulerOutput 的共享内存句柄
        shared_worker_lock: LockType,
        is_driver_worker: bool,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # ready_pipe: 子进程在完成初始化后，通过它把 READY 信号和 MQ handle 回传。
        reader, writer = context.Pipe(duplex=False)

        # death_pipe: 用于检测父进程退出。
        # 父进程保留 death_writer；一旦父进程退出，子进程读端会收到 EOF。
        death_reader, death_writer = context.Pipe(duplex=False)

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": (reader, writer),
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
        }
        # 在后台子进程中运行 worker_main / busy loop。
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=True,
        )

        proc.start()
        writer.close()
        # 父进程保留 death_writer。
        # 这样父进程一旦退出，子进程侧的 death_reader 就会收到 EOFError。
        return UnreadyWorkerProcHandle(proc, rank, reader, death_writer)

    @staticmethod
    def wait_for_response_handle_ready(
        handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle
    ) -> WorkerProcHandle:
        response_handle = handles["handle"]
        worker_response_mq: MessageQueue | None = None
        if len(response_handle.local_reader_ranks) > 0:
            worker_response_mq = MessageQueue.create_from_handle(response_handle, 0)
        peer_response_handles = handles["peer_response_handles"]
        peer_worker_response_mqs = [
            MessageQueue.create_from_handle(handle, -1)
            if handle.remote_subscribe_addr is not None
            else None
            for handle in peer_response_handles
        ]
        return WorkerProcHandle.from_unready_handle(
            proc_handle,
            worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
        )

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle],
    ) -> list[WorkerProcHandle]:
        e = Exception(
            "WorkerProc initialization failed due to "
            "an exception in a background process. "
            "See stack trace for root cause."
        )

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[WorkerProcHandle | None] = [None] * len(
            unready_proc_handles
        )
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    # 等待某个 WorkerProc 通过 ready_pipe 回报 READY。
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    idx = unready_proc_handle.rank % len(ready_proc_handles)
                    ready_proc_handles[idx] = WorkerProc.wait_for_response_handle_ready(
                        response, unready_proc_handle
                    )
                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # 无论成功失败，ready_pipe 这端都应关闭。
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self):
        self.worker.shutdown()
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def worker_main(*args, **kwargs):
        """worker 子进程入口。

        这个函数运行在后台子进程中，负责：
        1. 完成 WorkerProc 初始化
        2. 向父进程发送 READY 与消息队列句柄
        3. 进入 busy loop，不断接收 RPC 并执行
        """

        # 用于优雅退出的信号处理器。
        # 只抛一次 SystemExit，避免重复清理带来额外错误。
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                logger.debug(
                    "WorkerProc handling signal %d, raising SystemExit", signum
                )
                raise SystemExit()

        # 收到 SIGTERM 或 SIGINT 时都走统一退出逻辑。
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        # ready_pipe 的结构是 tuple[Connection, Connection]。
        reader, ready_writer = kwargs.pop("ready_pipe")
        death_pipe: Connection | None = kwargs.pop("death_pipe", None)
        shutdown_event = threading.Event()
        # 如果提供了 death_pipe，则启动线程监控父进程是否已经退出。
        if death_pipe is not None:

            def monitor_parent_death():
                try:
                    # 这里会一直阻塞，直到父进程退出并关闭 pipe。
                    death_pipe.recv()
                except EOFError:
                    # 父进程已退出，本 worker 也应尽快退出。
                    logger.info_once("Parent process exited, terminating worker")
                    # 通过 shutdown_event 通知 busy loop 结束。
                    shutdown_event.set()
                except Exception as e:
                    logger.warning("Death monitoring error: %s", e)

            death_monitor = Thread(
                target=monitor_parent_death, daemon=True, name="WorkerDeathMonitor"
            )
            death_monitor.start()

        try:
            reader.close()

            # 初始化 tracing。
            rank = kwargs.get("rank", 0)
            maybe_init_worker_tracer(
                instrumenting_module_name="vllm.worker",
                process_kind="worker",
                process_name=f"Worker_{rank}",
            )

            worker = WorkerProc(*args, **kwargs)
            assert worker.worker_response_mq is not None

            # 确认模型和消息队列都初始化完成后，再向父进程发送 READY。
            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,
                    "handle": worker.worker_response_mq.export_handle(),
                    "peer_response_handles": worker.peer_response_handles,
                }
            )

            # 等待各消息队列完成握手。
            # 顺序必须与 Executor 端保持一致，否则会死锁。
            if worker.rpc_broadcast_mq is not None:
                worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop(cancel=shutdown_event)

        except Exception:
            # 如果 busy loop 或初始化阶段抛出异常，executor 端通常会收到 FAILURE，
            # 进而触发整体关闭。
            # TODO(rob): 还需要更好处理 MQ 本身损坏的情况。

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            elif shutdown_event.is_set():
                logger.info("WorkerProc shutting down.")
            else:
                logger.exception("WorkerProc failed.")

            # 若某个 worker 出问题，父进程会向所有 worker 发送 SIGTERM。
            # 这里将该标记置位，避免后续再重复抛出 SystemExit，减少析构期异常噪音。
            shutdown_requested = True

        except SystemExit as e:
            # 收到退出信号时会走到这里。
            logger.warning("WorkerProc was terminated")
            # SystemExit 不能被吞掉，必须继续向外抛。
            raise e

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            # worker 退出 busy loop 后统一做清理。
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def enqueue_output(self, output: Any):
        """整理 worker 输出并写入 response MQ。

        如果输出是异常，会转换成 FAILURE 响应；否则按 SUCCESS 响应写回。
        """
        if isinstance(output, AsyncModelRunnerOutput):
            output = output.get_output()

        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)
        if (response_mq := self.worker_response_mq) is not None:
            response_mq.enqueue(result)

    def handle_output(self, output: Any):
        """处理 worker 输出。

        若启用了异步调度，则先交给异步输出线程处理；
        否则直接写入 worker_response_mq。
        """
        if self.use_async_scheduling:
            self.async_output_queue.put(output)
        else:
            self.enqueue_output(output)

    def async_output_busy_loop(self):
        """异步输出线程入口。"""
        while True:
            output = self.async_output_queue.get()
            self.enqueue_output(output)

    def worker_busy_loop(self, cancel: threading.Event | None = None):
        """多进程 worker 的主循环。"""
        assert self.rpc_broadcast_mq is not None
        while True:
            # 持续从广播 MQ 中取出父进程发来的 RPC 请求。
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
                cancel=cancel, indefinite=True
            )
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    # 若 method 是序列化后的可调用对象，则先反序列化再绑定 worker。
                    func = partial(cloudpickle.loads(method), self.worker)

                output = func(*args, **kwargs)
            except Exception as e:
                # Python 3.11 起，异常对象支持 add_note，可附带远端栈信息。
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                # 异常对象未必可序列化；这里只把它作为 FAILURE 响应往回传。
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(e)
                continue

            if output_rank is None or self.rank == output_rank:
                # 只有被指定为回包方的 rank 才需要返回结果。
                self.handle_output(output)

    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None:
        # 先检查并行组是否已经初始化。
        if not model_parallel_is_initialized():
            # 并行组还没初始化时，只能用默认进程名。
            set_process_title(name="Worker")
            decorate_logs("Worker")
            return

        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        pp_size = get_pp_group().world_size
        pp_rank = get_pp_group().rank_in_group
        pcp_size = get_pcp_group().world_size
        pcp_rank = get_pcp_group().rank_in_group
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        dcp_size = get_dcp_group().world_size
        dcp_rank = get_dcp_group().rank_in_group
        process_name = "Worker"
        if dp_size > 1:
            process_name += f"_DP{dp_rank}"
        if pp_size > 1:
            process_name += f"_PP{pp_rank}"
        if pcp_size > 1:
            process_name += f"_PCP{pcp_rank}"
        if tp_size > 1:
            process_name += f"_TP{tp_rank}"
        if dcp_size > 1:
            process_name += f"_DCP{dcp_rank}"
        if enable_ep:
            ep_rank = get_ep_group().rank_in_group
            process_name += f"_EP{ep_rank}"
        set_process_title(name=process_name)
        decorate_logs(process_name)


def set_multiprocessing_worker_envs():
    """配置多进程 worker 启动前应设置的环境变量。

    这个函数应由父进程在创建 worker 子进程之前调用。
    """

    _maybe_force_spawn()

    # 如果用户没有显式设置 OMP_NUM_THREADS，则主动收缩 Torch 的线程并行度。
    #
    # 这样做是为了降低 CPU 争用：如果每个 GPU worker 都默认拉满 CPU 线程，
    # 多进程场景下很容易互相抢核，容器环境里还可能因为 CPU quota 被放大成抖动。
    default_omp_num_threads = 1
    if (
        "OMP_NUM_THREADS" not in os.environ
        and (current_parallelism := torch.get_num_threads()) > default_omp_num_threads
    ):
        logger.warning(
            "Reducing Torch parallelism from %d threads to %d to avoid "
            "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
            "external environment to tune this value as needed.",
            current_parallelism,
            default_omp_num_threads,
        )
        os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
        torch.set_num_threads(default_omp_num_threads)
