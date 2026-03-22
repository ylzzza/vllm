# vLLM 运行逻辑梳理

这份笔记面向“先抓主线、再下钻源码”的阅读方式。  
当前 `vllm` 主线基本是 **V1 架构**，核心不是把请求硬拆成 `prefill` / `decode` 两套流程，而是围绕一个统一的调度循环展开：

1. 调度本轮要执行的请求和 token
2. 调用执行层跑模型
3. 采样新 token
4. 把结果回写调度器
5. 继续下一轮

---

## 1. 总体架构

可以把 `vllm` 看成 5 层：

```text
入口层 -> Engine 层 -> Core 层 -> 执行层 -> 输出层
```

对应关系大致如下：

```text
OpenAI API Server / LLM
        |
        v
 AsyncLLM / LLMEngine
        |
        v
  EngineCoreClient
        |
        v
     EngineCore
        |
        v
     Scheduler
        |
        v
 Executor -> Worker -> ModelRunner
        |
        v
  OutputProcessor
```

其中：

- `AsyncLLM` 是在线服务的前端壳
- `LLMEngine` 是离线路径的前端壳
- 两者底层都复用同一个 `EngineCore`

---

## 2. 在线服务调用链

在线模式从 OpenAI 兼容接口进入：

- [`vllm/vllm/entrypoints/openai/api_server.py`](vllm/vllm/entrypoints/openai/api_server.py)

服务初始化时会构建 `AsyncLLM`：

- [`vllm/vllm/v1/engine/async_llm.py`](vllm/vllm/v1/engine/async_llm.py)

`AsyncLLM` 初始化时主要创建：

- `InputProcessor`
- `OutputProcessor`
- `EngineCoreClient`
- 渲染与输出相关组件

请求进入后，主流程在 `AsyncLLM.add_request()`：

1. `InputProcessor.process_inputs(...)` 做输入规范化
2. `assign_request_id(...)` 把外部请求 id 转成内部 id
3. 把请求登记到 `OutputProcessor`
4. 把请求送到 `EngineCore`

对应代码：

- [`vllm/vllm/v1/engine/async_llm.py`](vllm/vllm/v1/engine/async_llm.py)
- [`vllm/vllm/v1/engine/input_processor.py`](vllm/vllm/v1/engine/input_processor.py)

流式输出时，`AsyncLLM.generate()` 会持续从每个请求自己的队列里取 `RequestOutput` 并向上游返回。后台还有 `_run_output_handler()` 循环，不断从 `EngineCore` 拉输出并交给 `OutputProcessor`。

---

## 3. 离线路径调用链

离线模式主要走 `LLMEngine`：

- [`vllm/vllm/v1/engine/llm_engine.py`](vllm/vllm/v1/engine/llm_engine.py)

它和 `AsyncLLM` 的区别主要在“前端表现形式”：

- `AsyncLLM` 面向异步流式服务
- `LLMEngine` 面向同步调用或批处理

但两者核心相同：

- 都会先用 `InputProcessor` 把输入转换成内部请求对象
- 都会把请求交给 `EngineCore`
- 都会通过 `OutputProcessor` 生成最终输出

所以你可以把它们理解成：

- `AsyncLLM` / `LLMEngine` 只是两种上层 API
- `EngineCore` 才是运行时主脑

---

## 4. EngineCore 是真正的主脑

`EngineCore` 初始化时会做几类关键工作：

1. 创建 `model_executor`
2. 估算可用显存并初始化 KV cache
3. 创建 `Scheduler`
4. 做 warmup / profiling / connector 初始化

对应代码：

- [`vllm/vllm/v1/engine/core.py`](vllm/vllm/v1/engine/core.py)

如果是多进程模式，前端不会直接调用 `EngineCore`，而是通过 `EngineCoreClient` 与后台 `EngineCoreProc` 通信：

- [`vllm/vllm/v1/engine/core_client.py`](vllm/vllm/v1/engine/core_client.py)
- [`vllm/vllm/v1/engine/core.py`](vllm/vllm/v1/engine/core.py)

这里通常有几种 client 形态：

- `InprocClient`
- `SyncMPClient`
- `AsyncMPClient`

也就是说，**前端看起来像是在调本地对象，但实际可能是通过 ZMQ 驱动后台核心进程**。

---

## 5. 单次 step 的闭环

`vllm` 最重要的一轮循环就在 `EngineCore.step()`，主线可以概括成：

```text
scheduler.schedule()
    -> model_executor.execute_model(...)
    -> sample_tokens(...)   # 某些路径会分开做
    -> scheduler.update_from_output(...)
```

关键文件：

- [`vllm/vllm/v1/engine/core.py`](vllm/vllm/v1/engine/core.py)
- [`vllm/vllm/v1/core/sched/scheduler.py`](vllm/vllm/v1/core/sched/scheduler.py)

这一步分别负责：

### 5.1 `scheduler.schedule()`

决定本轮：

- 哪些请求能继续跑
- 每个请求跑多少 token
- KV cache block 是否够用
- 不够时该 preempt 哪些请求

### 5.2 `model_executor.execute_model(...)`

把调度结果广播给执行层，真正发到 worker 上做一次模型执行。

### 5.3 `sample_tokens(...)`

对 logits 做采样，得到新 token，必要时也处理 grammar、spec decode 等逻辑。

### 5.4 `scheduler.update_from_output(...)`

把这轮执行结果回写到请求状态，包括：

- sampled tokens
- logprobs
- pooling 输出
- speculative decoding 的 accept/reject
- KV connector 相关回传信息

然后形成 `EngineCoreOutput` 给上层消费。

---

## 6. Scheduler 是 V1 最核心的设计点

V1 与旧架构最大的区别，是它**不再硬区分 `prefill` 和 `decode`**。

调度器核心看的是每个请求：

- `num_computed_tokens`
- `num_tokens_with_spec`

也就是：

- 已经算到哪里了
- 还剩多少 token 需要算

相关实现：

- [`vllm/vllm/v1/core/sched/scheduler.py`](vllm/vllm/v1/core/sched/scheduler.py)

所以 V1 的调度方式不是：

- 先把请求分到 `prefill batch`
- 再把请求分到 `decode batch`

而是统一抽象成：

- “每个请求还欠多少 token 没算完”
- “这轮还有多少 token budget”
- “KV cache 是否足够”

这种设计的意义是：

- 调度更统一
- 连续批处理更自然
- 更适合混合不同阶段、不同长度的请求
- 吞吐优化更集中在 token 级别完成

调度器内部重点维护的状态包括：

- `waiting`
- `running`
- `requests`
- `KVCacheManager`
- `EncoderCacheManager`

---

## 7. 执行层的职责边界

执行层可以拆成三段：

### 7.1 Executor

负责“怎么把任务分发到一组 worker 上”。

比如根据 backend 选择：

- multiprocess
- ray
- uni
- external launcher

相关代码：

- [`vllm/vllm/v1/executor/abstract.py`](vllm/vllm/v1/executor/abstract.py)
- [`vllm/vllm/v1/executor/multiproc_executor.py`](vllm/vllm/v1/executor/multiproc_executor.py)

### 7.2 Worker

负责“单个设备 / 单个 rank 的运行时管理”。

以默认 CUDA 路径为例，`GPUWorker` 主要负责：

- 初始化设备
- 初始化分布式环境
- 加载模型权重
- 初始化 KV cache
- warmup
- 执行本轮模型计算

相关代码：

- [`vllm/vllm/v1/worker/gpu_worker.py`](vllm/vllm/v1/worker/gpu_worker.py)

### 7.3 ModelRunner

负责“把本轮调度结果组织成一次可执行的 forward / sampling”。

它通常维护：

- persistent batch
- request states
- 输入 buffer
- sampler
- attention metadata
- KV cache / encoder cache

关键路径：

- 更新 batch 与状态
- 准备模型输入
- 执行 forward
- 采样 token
- 更新内部状态

相关代码：

- [`vllm/vllm/v1/worker/gpu_model_runner.py`](vllm/vllm/v1/worker/gpu_model_runner.py)

一句话概括：

- `Executor` 管“多 worker 编排”
- `Worker` 管“单设备生命周期”
- `ModelRunner` 管“单轮模型怎么跑”

---

## 8. 输出层是怎么回到用户的

模型结果不会直接原样返回给用户，而是先经过 `OutputProcessor`：

- [`vllm/vllm/v1/engine/output_processor.py`](vllm/vllm/v1/engine/output_processor.py)

这一层主要负责：

- detokenize
- 组装 logprobs
- stop string 处理
- 生成 `RequestOutput`

在线模式下的完整输出链路可以理解成：

```text
EngineCoreOutput
    -> OutputProcessor
    -> per-request queue
    -> AsyncLLM.generate()
    -> HTTP streaming
```

离线模式下则通常由 `LLMEngine.step()` 直接返回处理好的输出列表。

---

## 9. `vllm-ascend` 是怎么接进来的

如果把上面的主线当成“标准 `vllm` 运行时”，那么 `vllm-ascend` 主要是在**平台与执行层**做适配，而不是重写整个调度框架。

你现在打开的几个文件，对应关系可以这样看：

- [`vllm-ascend/vllm_ascend/platform.py`](vllm-ascend/vllm_ascend/platform.py)
  - 定义 Ascend 平台能力、平台相关行为
- [`vllm-ascend/vllm_ascend/worker/model_runner_v1.py`](vllm-ascend/vllm_ascend/worker/model_runner_v1.py)
  - 适配 V1 路径上的 model runner 执行逻辑
- [`vllm-ascend/vllm_ascend/worker/v2/model_runner.py`](vllm-ascend/vllm_ascend/worker/v2/model_runner.py)
  - Ascend 上更具体的执行路径实现

所以可以把两者关系概括为：

```text
vLLM 提供：
- 请求接入
- 调度器
- EngineCore
- 通用执行框架

vllm-ascend 提供：
- Ascend 平台注册
- Ascend worker / model runner
- NPU 执行细节和能力适配
```

换句话说：

- `vllm` 决定“什么时候跑、跑哪些请求、怎么调度”
- `vllm-ascend` 决定“在 Ascend 上这一轮具体怎么执行”

---

## 10. 推荐阅读顺序

如果你想顺着源码建立全局认识，推荐按这个顺序读：

1. [`vllm/vllm/entrypoints/openai/api_server.py`](vllm/vllm/entrypoints/openai/api_server.py)
2. [`vllm/vllm/v1/engine/async_llm.py`](vllm/vllm/v1/engine/async_llm.py)
3. [`vllm/vllm/v1/engine/input_processor.py`](vllm/vllm/v1/engine/input_processor.py)
4. [`vllm/vllm/v1/engine/core_client.py`](vllm/vllm/v1/engine/core_client.py)
5. [`vllm/vllm/v1/engine/core.py`](vllm/vllm/v1/engine/core.py)
6. [`vllm/vllm/v1/core/sched/scheduler.py`](vllm/vllm/v1/core/sched/scheduler.py)
7. [`vllm/vllm/v1/executor/abstract.py`](vllm/vllm/v1/executor/abstract.py)
8. [`vllm/vllm/v1/worker/gpu_worker.py`](vllm/vllm/v1/worker/gpu_worker.py)
9. [`vllm/vllm/v1/worker/gpu_model_runner.py`](vllm/vllm/v1/worker/gpu_model_runner.py)
10. [`vllm/vllm/v1/engine/output_processor.py`](vllm/vllm/v1/engine/output_processor.py)

如果你的重点是 Ascend 适配，再接着读：

1. [`vllm-ascend/vllm_ascend/platform.py`](vllm-ascend/vllm_ascend/platform.py)
2. [`vllm-ascend/vllm_ascend/worker/model_runner_v1.py`](vllm-ascend/vllm_ascend/worker/model_runner_v1.py)
3. [`vllm-ascend/vllm_ascend/worker/v2/model_runner.py`](vllm-ascend/vllm_ascend/worker/v2/model_runner.py)

---

## 11. 一句话总结

`vllm` 的运行主线可以浓缩成一句话：

> 上层 API 把请求送进 `EngineCore`，`Scheduler` 决定本轮跑什么，执行层完成模型计算与采样，`OutputProcessor` 再把结果整理成用户可见输出。

如果再缩成一个最核心的点，那就是：

> **理解 `EngineCore.step()` 与 `Scheduler.schedule()`，就抓住了 `vllm` V1 的主线。**
