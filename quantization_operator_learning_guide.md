# 量化算子学习项目推荐

这份笔记按“从零开始学 GPU / 量化算子”的标准整理，优先考虑：

- 上手门槛低
- 教程和示例比较完整
- 代码可读性较好
- 能逐步过渡到 `vLLM` 相关实现

## 最推荐的学习顺序

建议按下面这条路线学：

`Triton 官方教程 -> GemLite -> Marlin -> TorchAO / llm-compressor -> FlashInfer`

如果你的目标是尽快进入 `vLLM` 量化相关代码，这条路线的性价比最高。

## 第一阶段：先学基础 kernel

### 1. Triton 官方教程

最适合作为第一站。

为什么适合入门：

- 教程体系完整，能从最基础的向量加法一路看到 `matmul`、`softmax`、`attention`
- 写法比 CUDA 更容易读，适合先建立线程块、tile、memory access 的直觉
- 官方教程里已经有 low-bit / block-scaled matmul 相关内容，能自然过渡到量化算子

建议先看的内容：

- Vector Add
- Softmax
- Matrix Multiplication
- Fused Attention
- Block-Scaled MatMul

链接：

- https://github.com/triton-lang/triton
- https://triton-lang.org/main/getting-started/tutorials/index.html
- https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html

### 2. GemLite

这是第一个值得认真读的“量化线性层 / 低比特 matmul”项目。

为什么适合入门：

- 重点非常聚焦，就是 low-bit linear / matmul
- 基于 Triton，学习成本明显低于直接啃 CUDA + CUTLASS
- 支持 `8/4/2/1-bit` 权重量化，能直接接触 packing、dequant、group size 这些核心概念
- 跟 `vLLM V1`、`TorchAO`、`SGLang` 有实际集成，离生产场景不远

看这个项目时重点关注：

- weight packing 方式
- scale / zero-point 如何组织
- kernel 内部 dequant 的位置
- 为什么有的实现更偏 `GEMV`，有的更偏 `GEMM`

链接：

- https://github.com/mobiusml/gemlite

## 第二阶段：开始看更真实的量化 kernel

### 3. Marlin

这是很适合作为“第一个 CUDA 级量化 kernel 项目”的仓库。

为什么适合这个阶段：

- 目标明确，主要聚焦 `FP16 x INT4`
- 仓库体量比很多工业项目更小，阅读负担相对可控
- 很多后续 weight-only INT4 实现都受它影响，属于经典案例

建议重点看：

- quantized weight 的排布方式
- 为什么 kernel 会围绕特定 tile 形状设计
- dequant 与 matmul 的融合方式
- 吞吐瓶颈是算力、带宽还是寄存器压力

链接：

- https://github.com/IST-DASLab/marlin

### 4. bitsandbytes

如果你想补一层“PyTorch 里量化线性层到底怎么被模型调用”的理解，这个项目很好。

为什么值得看：

- 提供 `Linear8bitLt`、`Linear4bit` 这类直接可用的模块
- 能帮助你把“底层算子”与“模型层替换”联系起来
- 对理解推理框架里的量化 layer 封装方式很有帮助

链接：

- https://github.com/bitsandbytes-foundation/bitsandbytes

## 第三阶段：过渡到 vLLM 相关实现

### 5. TorchAO

它不只是 kernel 项目，更像量化框架，但非常适合补齐体系化认知。

为什么值得学：

- 能系统理解 `int4`、`fp8`、weight-only quantization、activation quantization 这些概念
- 能看到 PyTorch 生态里 quantized tensor / module 是怎么组织的
- 与 `vLLM` 已有联动，便于后续对照 `vLLM` 量化后端实现

建议重点看：

- low-bit linear
- quantization config 定义
- tensor layout / packing 抽象
- runtime dispatch

链接：

- https://github.com/pytorch/ao

### 6. llm-compressor

如果你要从“量化算法 / 模型导出”走向 `vLLM` 实际接入，这个项目非常关键。

为什么值得看：

- 直接对接 `vLLM` 生态
- 能看到 GPTQ、AWQ、SmoothQuant、FP8 这类方案怎么落到实际模型上
- 有助于区分“量化算法层”和“kernel 层”分别负责什么

链接：

- https://github.com/vllm-project/llm-compressor
- https://docs.vllm.ai/en/stable/features/quantization/

## 暂时不建议作为第一站的项目

### FlashInfer

非常强，但更适合你已经懂一些 kernel 之后再看。

原因：

- 更偏成熟工业内核库
- 涉及 attention、GEMM、MoE、paged KV cache 等多个方向
- 信息密度高，刚入门时容易被实现细节淹没

链接：

- https://github.com/flashinfer-ai/flashinfer

### CUTLASS

这是很重要的参考库，但不适合拿来做第一份量化算子代码阅读材料。

原因：

- 模板和抽象层比较重
- 更适合作为“查标准做法”和“对照工业实现”的工具书

链接：

- https://github.com/NVIDIA/cutlass

### OmniServe / QServe

更适合你已经掌握量化 kernel 基础后，再学习“量化方案 + serving system co-design”。

链接：

- https://github.com/mit-han-lab/omniserve

## 如果只选 3 个项目

如果现在时间有限，只看这 3 个：

1. Triton 官方教程
2. GemLite
3. Marlin

这 3 个组合最适合建立下面这条能力链：

`写简单 kernel -> 读懂低比特 matmul -> 理解真实量化推理实现`

## 一个务实的学习方法

不要一开始就试图“全面理解量化”。更有效的方法是按这个顺序拆：

1. 先能写出一个最简单的 Triton `matmul`
2. 再理解 weight-only quantization 的基本数据结构
3. 然后看 low-bit weight 怎么 pack
4. 再看 dequant 是在 kernel 外做还是在 kernel 内做
5. 最后再去读 `vLLM` 里的 quantization backend

## 针对 vLLM 的建议

如果你的最终目标是给 `vLLM` 加量化算子或读懂现有实现，建议优先补这些关键词：

- `weight-only quantization`
- `group size`
- `per-channel / per-group scales`
- `packed layout`
- `fused dequant + matmul`
- `GEMV vs GEMM`
- `FP8 / block scaling`

等这些概念建立起来，再去看 `vLLM` 的量化后端和相关 kernel，理解会快很多。
