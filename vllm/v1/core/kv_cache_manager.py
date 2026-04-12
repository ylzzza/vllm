# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """
    KVCacheManager 的分配结果。

    它充当 Scheduler 与 KVCacheManager 之间的接口层，用来把
    KVCacheManager 的内部数据结构封装起来，避免 Scheduler 直接依赖
    其内部表示。
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` 表示第 `i` 个 kv cache group 中的第 `j` 个 token block。

    这里不把“token block”作为最外层维度，是因为那会隐含一个前提：
    所有 kv_cache_group 拥有相同数量的 block。这个前提目前通常成立，
    但如果以后不同 kv_cache_group 支持不同的 block_size，这种表示就会失效。

    单个 KVCacheBlocks 在每个 group 上的表示方式可以是：
    - `list[KVCacheBlock]`：该 group 下存在一个或多个 block
    - 空 tuple：该请求在该 group 下没有任何 block
      （KVCacheManager 中会预先构造一个空 KVCacheBlocks，以减少 GC 开销）
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """将两个 KVCacheBlocks 实例按 group 逐一拼接。"""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        将 KVCacheBlocks 转换成 block_id 视图。

        返回：
            tuple[list[int], ...]: 一个由 list 构成的 tuple，其中：
                - 最外层 tuple 对应 KV cache groups
                - 每个内部 list 保存该 group 中各个 block 的 block_id
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """返回当前 KVCacheBlocks 中尚未计算 hash 的 block_id。"""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def new_empty(self) -> "KVCacheBlocks":
        """创建一个与当前 group 结构一致、但不包含任何 block 的新对象。"""
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    """面向 Scheduler 的 KV cache 管理外观层。

    这一层本身不实现复杂的 block 分配算法，而是把 Scheduler 在运行时
    最关心的几类操作封装成稳定接口：

    1. `get_computed_blocks()`：查询本地 prefix cache 已命中的完整 block
    2. `allocate_slots()`：为本轮要继续使用或新增计算的 token 申请 block
    3. `cache_blocks()`：把本轮已经“稳定可复用”的前缀写入缓存
    4. `free()`：在请求结束或被抢占时释放 block

    真正的 block 布局、滑窗裁剪、prefix cache 命中、引用计数和 block pool
    管理，都下沉给 coordinator / block_pool 处理。
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: 未来可以把 prefix cache 统计是否启用做成独立配置。
        # 目前仍然跟随 log_stats 开关，但后续可能还会暴露更细粒度的统计选项。
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        # coordinator 是真正执行“block 级别调度/缓存/回收”的组件。
        # KVCacheManager 负责把 Scheduler 的 request/token 语义转换成
        # coordinator 能直接处理的 block 级操作。
        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # 预先构造一个“不含任何 block”的 KVCacheBlocks，调用方应尽量通过
        # create_kv_cache_blocks() 复用它，避免频繁创建空对象带来的 GC 开销。
        #
        # 这里使用嵌套 tuple，确保这个空对象是不可变的。
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

    @property
    def usage(self) -> float:
        """返回 KV cache 当前使用率。

        返回：
            KV cache 使用率，范围在 0.0 到 1.0 之间。
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """获取并重置 prefix cache 统计信息。

        返回：
            当前 prefix cache 统计；若未启用日志，则返回 None。
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """查询某个请求已经命中的完整缓存 block。

        注意：这里返回的“已计算 block”必须是完整 block，不能是半个 block。

        参数：
            request: 要查询的请求。

        返回：
            一个二元组，包含：
                - 该请求已命中的 block 列表
                - 已命中的 token 数
        """
        # 如果关闭了 prefix caching，或者请求显式要求跳过 KV cache 读取，
        # 那就直接视为“没有任何本地缓存命中”。
        # 典型场景包括：请求 prompt logprobs，或某些 pooling 模型路径。
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # 即使整个 prompt 都命中缓存，也必须至少重算最后一个 token，
        # 因为只有这样才能拿到当前位置对应的 logits。因此这里把最大命中长度
        # 限制为 prompt_length - 1。
        #
        # 由于 allocate_slots() 要求 num_computed_tokens 与 block 对齐，
        # 这有时会导致“为了重算最后一个 token，不得不重算整个 block”。
        # 未来如果放宽这个约束，性能还有提升空间。
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        """为请求追加本轮需要的 KV slots / blocks。

        参数：
            request: 需要分配 slot 的请求。
            num_new_tokens: 本轮要为其分配并真正参与计算的新 token 数量。
            num_new_computed_tokens: 本轮刚刚通过 prefix cache 命中的“本地”
                token 数，不包含外部 connector 提供的 token。
            new_computed_blocks: 上述新命中的缓存 block，按 kv cache group
                组织成 tuple。
            num_lookahead_tokens: 需要额外预留的 speculative token 数量，
                用于 EAGLE 等带 kv-cache 的 spec decode proposer。
            num_external_computed_tokens: 这些 token 的 KV 不在 vLLM 本地缓存，
                但已经由 connector 持有。
            delay_cache_blocks: 是否暂时跳过 cache 提交。典型用于 P/D 场景：
                这些 block 会参与一次尚未完成的远端 KV 传输，因此只能先占位。
            num_encoder_tokens: encoder-decoder 模型中，为 cross-attention
                额外分配的 encoder token 数量，例如 Whisper。对 decoder-only
                模型，这个值应为 0。

        Token/Block 布局示意：
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | 由 vLLM 或 connector 提供的前缀缓存    |
        | 如果这些 token 已落在滑窗之外，则可以   |
        | 安全移除                                |
        ----------------------------------------------------------------------
        |   < 由 vLLM 缓存 >      | 不在 vLLM     |
                                  | 缓存中，但由  |
        | ref_cnt  | ref_cnt 尚未 | connector     |
        | 已增加   | 增加         | 持有          |
        ----------------------------------------------------------------------
        ```

        缩写说明：

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens，由 connector 持有
        new       = num_new_tokens，包含尚未验证的 draft token
        lookahead = num_lookahead_tokens
        ```

        注意：如果 `new` 中同时包含“已验证”和“未验证”的 draft token，
        那么真正写入缓存时只能提交已经确定有效的那部分。因此这里会把可缓存
        token 数限制到 `request.num_tokens`，避免把未来可能被拒绝的 draft
        token 提前写进 prefix cache。

        运行逻辑可以分成三个阶段：
        1. 先处理旧前缀 `comp`：
           释放不再需要的 block，并检查剩余空闲 block 是否足够。
           若不足则直接返回 None。
        2. 再处理“已知前缀” `comp + new_comp + ext_comp`：
           - 继续释放不再参与注意力计算的 block，例如滑窗外的 block
           - 为处在滑窗内、但仅存在于外部 connector 中的 `ext_comp`
             token 预留本地 block
        3. 最后为本轮真正要计算的 token，也就是 `new + lookahead`
           分配新 block

        返回：
            本轮新分配到的 block 列表；如果资源不足则返回 None。
        """
        # 异步加载远端 KV 时，可能当前并没有新的 token 要计算，
        # 但仍然需要为“外部已计算 token”先占好本地 slot。
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # 先把“当前已算过的 token”与“本轮新命中的本地 prefix token”合并，
        # 得到本地视角下已经就绪的 token 数。
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        # total_computed_tokens 则再加上外部 connector 已持有 KV 的 token。
        # 这表示本轮开始前，对主模型来说已经可直接复用的上下文总长度。
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        # 主模型这一轮真正会看到的 token 总长度 = 已有上下文 + 本轮新增 token。
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        # 如果还要给 spec decode 预留 lookahead，就需要更多 slot，但同样不能
        # 超过模型最大长度。
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens,
            self.max_model_len,
        )

        # 先释放在注意力计算里已经不会再访问的 block，例如滑窗之外的旧 block。
        # 即便后面因为空闲 block 不足而无法调度，本次裁剪依然是有价值的。
        # 把这一步放在分配新 block 前面，也能减少后续可能发生的驱逐。
        self.coordinator.remove_skipped_blocks(
            request.request_id, total_computed_tokens
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # 空闲 block 不够，本轮无法继续调度该请求。
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # 先把“本轮新增命中的 prefix block”和“外部已有 KV 的 block 占位”
            # 接到请求的 block 表后面，再进入新 block 分配。
            # 这样可以保证后续计算 `allocate_new_blocks()` 时，看到的是完整的
            # 逻辑前缀布局，而不是只看到旧 block。
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # P/D 场景下，如果这些 block 还要等待远端 KV 接收完成，
        # 这里只能先返回分配结果，不能立刻把它们提交进可复用缓存。
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # 理想上，我们希望把“本地已算 + 外部已算 + 本轮新算”的 token 都提交到
        # 缓存里；但其中可能包含尚未最终确认的 draft token，它们未来可能被拒绝。
        # 因此只能把 token 上限截到 request.num_tokens，确保进入 prefix cache
        # 的都是已经稳定下来的 token。
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def free(self, request: Request) -> None:
        """释放某个请求占用的 block。

        释放时会按逆序处理，这样在启用缓存时，尾部 block 会优先被淘汰。

        参数：
            request: 要释放 block 的请求。
        """
        self.coordinator.free(request.request_id)

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """移除已经不再需要的 block，并用 null_block 替换其位置。

        参数：
            request_id: 请求 ID。
            total_computed_tokens: 总已计算 token 数，包含本地已计算 token 和
                外部已计算 token。
        """
        self.coordinator.remove_skipped_blocks(request_id, total_computed_tokens)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """按 block_id 从 prefix cache 中驱逐 block。

        参数：
            block_ids: 要从缓存中驱逐的 block_id 集合。
        """
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """重置 prefix cache。

        这个函数可用于 RLHF 等流程，在模型权重更新后使 prefix cache 失效，
        也可用于基准测试时重置 prefix cache 状态。

        返回：
            bool: 若 prefix cache 成功重置则返回 True，否则返回 False。
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """计算每个 kv cache group 的公共前缀 block 数量。

        这个函数会选取一个正在运行的请求，并遍历它的 block。
        若一个 block 被所有“已经分配了 KV cache 的请求”共享，也就是它的
        `ref_cnt` 等于 `req_to_blocks` 中条目的总数，那么它就被视为公共前缀
        block。

        注意：拥有已分配 KV cache 的请求数量，**大于等于** 当前调度步中
        真正被调度的请求数量。因为“拥有 KV cache”只表示：
        1. 该请求尚未结束
        2. 它持有的 block 还没有被释放

        所有被调度的请求一定都有已分配 KV cache，但反过来并不成立：
        某些请求虽然持有 KV cache，却可能没有在当前这一步真正参与调度。

        这会带来一个边界情况：即使当前所有“已调度请求”都共享某个公共前缀，
        公共前缀 block 数仍然可能返回 0。原因是还可能存在某些“未调度但仍持有
        KV cache 的请求”，它们并不共享这个前缀。目前这个情况不太容易精确
        检测，因此这里会保守地返回 0。

        参数：
            running_request_id: 任意一个 running 请求的 request ID，用它来
                识别公共前缀 block。

        返回：
            list[int]: 每个 kv cache group 对应的公共前缀 block 数量。
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """从 block pool 中取出 KV cache 事件。

        返回：
            KV cache 事件列表。
        """
        return self.block_pool.take_events()

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """获取某个请求当前持有的 block。"""
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """获取某个请求当前持有的 block_id。"""
        return self.get_blocks(request_id).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """在启用缓存时，把请求对应的 block 提交到 cache。

        参数：
            request: 要提交缓存的请求。
            num_computed_tokens: 已计算 token 数，包含已经缓存的 token 和
                本次准备写入缓存的 token。
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # 只有在至少一个 group 非空时才创建新对象；空结果统一复用预构造实例。
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    def new_step_starts(self) -> None:
        """在一个新的调度步开始时调用。"""
        self.coordinator.new_step_starts()
