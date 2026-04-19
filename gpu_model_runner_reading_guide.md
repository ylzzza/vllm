# `gpu_model_runner.py` 阅读建议

这份笔记面向“先抓主线，再下钻细节”的阅读方式。  
`vllm/v1/worker/gpu_model_runner.py` 很大，里面既有主执行逻辑，也揉进了很多优化分支。第一次阅读时，不建议从头到尾平均用力。

更好的方式是：

1. 先理解这个类在整个 V1 执行链路中的定位
2. 再抓住一次 step 的主调用链
3. 最后再回头看 speculative decoding、multimodal、CUDA graph、KV sharing 这些优化分支

---

## 1. 这个文件的定位

`GPUModelRunner` 不是模型本体，而是 worker 侧真正驱动 GPU 执行的一层。

你可以把它理解成 scheduler 与模型 forward 之间的“翻译层”：

1. 接收 scheduler 产出的本轮执行计划
2. 把请求状态整理成持久化 batch
3. 再把持久化 batch 转成本轮 forward 真正要吃的张量和 metadata
4. 执行模型
5. 采样 token
6. 把结果写回请求状态，供下一轮继续

核心类位置：

- [`vllm/v1/worker/gpu_model_runner.py`](vllm/vllm/v1/worker/gpu_model_runner.py)

相关入口：

- `class GPUModelRunner`
- `execute_model()`
- `sample_tokens()`

---

## 2. 不要一开始就读完整个文件

这个文件超过 6000 行，而且混合了很多功能：

- 普通文本生成
- pooling 模型
- pipeline parallel
- data parallel
- context parallel
- speculative decoding
- multimodal encoder
- CUDA graph
- LoRA
- KV transfer / EC transfer
- hybrid attention + mamba

所以第一次阅读时，不建议把它当成“一个线性函数”来看。  
更实用的做法是先盯住几组关键入口：

- `__init__()`
- `load_model()`
- `initialize_kv_cache()`
- `execute_model()`
- `sample_tokens()`

对应位置：

- [`vllm/v1/worker/gpu_model_runner.py`](vllm/vllm/v1/worker/gpu_model_runner.py)

建议优先搜索这些函数名，而不是顺序翻页。

---

## 3. 先理解两个核心状态对象

在读 `GPUModelRunner` 之前，最好先搞清楚两个状态容器：

- `CachedRequestState`
- `InputBatch`

对应文件：

- [`vllm/v1/worker/gpu_input_batch.py`](vllm/vllm/v1/worker/gpu_input_batch.py)

### 3.1 `CachedRequestState`

它表示“单个请求跨 step 持续存在的状态”，典型内容包括：

- `req_id`
- `prompt_token_ids`
- `output_token_ids`
- `block_ids`
- `num_computed_tokens`
- `sampling_params`
- 多模态特征
- RoPE 位置
- LoRA / pooling 相关状态

你可以把它理解成请求级数据库记录。

### 3.2 `InputBatch`

它表示“当前 worker 维护的持久化 batch 视图”，是一个按 `req_index` 排列的运行时容器。  
这里保存的不是“一次性构造完就扔掉”的 batch，而是跨 step 复用、不断增删改的结构。

典型内容包括：

- `req_id_to_index`
- `token_ids_cpu`
- `num_prompt_tokens`
- `num_computed_tokens_cpu`
- `block_table`
- sampling metadata
- speculative decoding 状态
- async scheduling 相关缓存

可以简单理解为：

- `requests` 更像请求级长期状态
- `input_batch` 更像当前 GPU 执行视角下的批状态

如果这两个对象没看懂，后面 `GPUModelRunner` 很多代码会显得杂乱。

---

## 4. 初始化阶段要看什么

### 4.1 `__init__()`

`GPUModelRunner.__init__()` 主要做的是搭运行时框架，而不是直接执行模型。

它大致做了这些事：

1. 保存各种 config 引用
2. 初始化 sampler、spec decode drafter 等组件
3. 创建持久化 `InputBatch`
4. 预分配大量 CPU/GPU buffer
5. 初始化异步拷贝 stream / event
6. 准备 multimodal、LoRA、KV sharing、CUDA graph 相关状态

阅读时重点关注两类成员：

- “请求与 batch 状态”成员  
  例如 `self.requests`、`self.input_batch`
- “高频复用 buffer”成员  
  例如 `self.input_ids`、`self.positions`、`self.seq_lens`

这会帮助你理解后面为什么很多函数只是在“填 buffer”，而不是临时创建新张量。

### 4.2 `load_model()`

这个函数主要处理：

1. 加载主模型
2. 按需叠加 LoRA
3. 加载 drafter
4. 处理与模型结构绑定的能力，比如 aux hidden states

它负责让“模型对象本身”就绪。

### 4.3 `initialize_kv_cache()`

这个函数负责让“执行环境”就绪。

主要工作：

1. 修正 KV cache 配置
2. 初始化 attention backend
3. 计算 kernel block size
4. 初始化 metadata builder
5. 必要时重建 `InputBatch`
6. 分配并绑定真正的 KV cache tensor

如果说 `load_model()` 解决的是“模型权重和结构”，那 `initialize_kv_cache()` 解决的是“这套模型怎么在当前 backend 上跑起来”。

---

## 5. 一次 step 的主链路

阅读这个文件时，最重要的是抓住一次 step 的主流程：

```text
execute_model()
    -> _update_states()
    -> _prepare_inputs()
    -> _build_attention_metadata()
    -> _preprocess()
    -> _model_forward()
    -> 暂存 ExecuteModelState

sample_tokens()
    -> _sample()
    -> _update_states_after_model_execute()
    -> _bookkeeping_sync()
    -> 返回 ModelRunnerOutput
```

也就是说，`forward` 和 `sample` 在 V1 里是拆开的。

这点很重要，因为很多异步优化、PP 广播、spec decode 都依赖这种拆分。

---

## 6. `execute_model()` 建议怎么读

`execute_model()` 是整个文件最重要的函数。  
第一次阅读时，建议只看它的高层骨架，不要立刻钻进每个 helper。

可以把它分成 5 步。

### 6.1 第一步：更新本地状态

调用：

- `_update_states()`

它做的事情包括：

- 删除 finished 请求
- 把本轮未调度请求从 `InputBatch` 中移除
- 加入新请求 / 恢复请求
- 更新旧请求的 token / block table / spec token
- `condense()` 压紧 batch
- `refresh_metadata()` 刷新 sampling metadata

这一层本质是在把 scheduler 输出同步到本地持久状态。

### 6.2 第二步：准备本轮输入

调用：

- `_prepare_inputs()`

这里会生成：

- `input_ids`
- `positions`
- `query_start_loc`
- `seq_lens`
- `logits_indices`
- speculative decoding metadata

这一步非常关键，因为它把“按请求组织的状态”转换成了“按 token 展平的模型输入”。

如果你想深读这个文件，`_prepare_inputs()` 是最值得花时间的地方。

### 6.3 第三步：构建 attention metadata

调用：

- `_build_attention_metadata()`

这一层是 attention backend 相关逻辑的核心入口。  
它会基于 block table、slot mapping、seq len、query start loc 等信息构造 backend 所需 metadata。

这一段如果你暂时不研究 backend 细节，可以先把它理解成：

- “把当前 batch 的 attention 上下文包装成 kernel 可直接消费的结构”

### 6.4 第四步：补充 forward 其它输入

调用：

- `_preprocess()`

这里会处理：

- multimodal encoder
- prompt embeds
- encoder-decoder 的 encoder outputs
- pipeline parallel 的 `intermediate_tensors`

如果你现在只想看文本生成，可以先把这部分当成“扩展输入准备层”。

### 6.5 第五步：执行 forward

调用：

- `_model_forward()`

然后根据不同场景分流：

- 非最后一个 PP rank 返回 `IntermediateTensors`
- pooling 模型走 `_pool()`
- 生成模型计算 `logits`
- 最终把结果塞进 `ExecuteModelState`

这里有一个关键点：  
`execute_model()` 对生成模型并不直接返回最终 token，而是把 `logits + hidden_states + metadata` 暂存起来，交给 `sample_tokens()` 继续处理。

---

## 7. `sample_tokens()` 建议怎么读

`sample_tokens()` 是执行链路的后半段。

它的典型流程是：

1. 取出 `execute_model_state`
2. 如果有 grammar，先对 logits 打 mask
3. 调 `_sample()` 做采样
4. 调 `_update_states_after_model_execute()` 维护 hybrid/spec 状态
5. 必要时继续生成 draft token
6. 调 `_bookkeeping_sync()` 把 sampled token 写回请求状态
7. 组装 `ModelRunnerOutput`

阅读时重点注意两层含义：

- “采样”不只是从 logits 里选 token，还可能涉及 grammar、penalty、spec decode
- “bookkeeping” 不是简单收尾，而是在更新下一轮会继续使用的状态

所以这里不是输出层的附属逻辑，而是调度闭环的一部分。

---

## 8. 异步调度为什么重要

这个文件里一个很容易忽略但非常关键的点是：  
很多输出不会立刻同步回 CPU，而是尽量异步处理。

代表类：

- `AsyncGPUModelRunnerOutput`

它的意义是：

- forward / sample 结束后，先异步把 sampled tokens、logprobs 拷到 CPU
- 不在当前路径上阻塞等待
- 等真正需要消费输出时，再同步

这样可以让：

- GPU 到 CPU 的 DMA
- 下一轮输入准备
- 一些后续 bookkeeping

尽可能重叠起来。

如果你看到代码里有很多 `stream`、`event`、`non_blocking=True`，核心都可以放到这个目标下理解。

---

## 9. 第一次阅读时可以先忽略什么

为了先抓主线，下面这些分支第一次可以先略过：

- M-RoPE / XD-RoPE 的细节
- Mamba 相关缓存布局
- KV sharing fast prefill
- routed experts capturer
- encoder cache / EC transfer
- KV transfer connector
- ubatching / DBO 细节
- CUDA graph capture 细节

不是说这些不重要，而是它们都建立在主链路已经理解的前提上。  
如果一开始就钻这些细节，很容易失去整体感。

---

## 10. 推荐阅读顺序

建议按下面顺序读：

1. [`vllm/v1/worker/gpu_input_batch.py`](vllm/vllm/v1/worker/gpu_input_batch.py)  
   先看 `CachedRequestState` 和 `InputBatch`

2. [`vllm/v1/worker/gpu_model_runner.py`](vllm/vllm/v1/worker/gpu_model_runner.py)  
   先看 `GPUModelRunner.__init__()`

3. `load_model()`

4. `initialize_kv_cache()`

5. `execute_model()`

6. 顺着 `execute_model()` 跳进去看：
   - `_update_states()`
   - `_prepare_inputs()`
   - `_build_attention_metadata()`
   - `_preprocess()`

7. 回来看 `sample_tokens()`

8. 最后再按兴趣读：
   - speculative decoding
   - multimodal
   - CUDA graph
   - LoRA
   - connector / transfer

---

## 11. 如果你要继续深挖，优先深读哪几块

如果你已经理解主线，后续最值得深读的 3 块是：

### 11.1 `_prepare_inputs()`

这是“从请求状态到模型输入”的核心转换层。  
很多你以后想调试的问题，最终都会落到这里：

- token 展平顺序对不对
- `positions` 对不对
- `logits_indices` 为什么是这样
- async sampled tokens 如何回填

### 11.2 `_update_states()`

这是“scheduler 输出如何落地到本地持久 batch”的关键层。  
如果你想理解连续批处理、请求恢复、请求删除、流式会话更新，这一段很关键。

### 11.3 `sample_tokens()` + `_bookkeeping_sync()`

这是“本轮执行结果如何进入下一轮状态”的关键层。  
你可以把它理解成执行闭环的“回写半边”。

---

## 12. 一句总结

阅读 `gpu_model_runner.py` 时，最重要的认知不是“它在调用哪个 kernel”，而是：

**它在维护一个跨 step 持续存在的请求状态系统，并把 scheduler 的决策翻译成一次次可执行的 GPU forward。**

如果你始终带着这个视角去读，很多看起来分散的 helper 函数都会变得容易理解。
