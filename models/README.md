# 示例模型库

在画布工具栏点「导入」，选择下面的 JSON 文件即可载入画布查看/编辑/训练。每个文件都是完整的 Graph IR 拓扑（格式说明见 `docs/拓扑Schema与算子贡献指南.md`），配合 `docs/大模型知识教程.md` 使用效果最佳。

| 文件 | 结构 | 配套数据集 | 参数量 | 教学重点 |
|---|---|---|---|---|
| `01-mnist-mlp.json` | Flatten→Linear→ReLU→Linear | mnist | 101,770 | 线性层、为什么要激活、输出端别接 Softmax |
| `02-mnist-cnn.json` | 双 Conv2d+ReLU+MaxPool+全连接头 | mnist | 206,922 | 卷积形状公式 (H+2p−k)/s+1、权重共享 |
| `03-mnist-resnet.json` | Conv→ReLU→ResNetBlock→MaxPool→全连接 | mnist | 81,610 | 残差连接、1×1 卷积做 skip 形状适配 |
| `04-text-lstm.json` | Embedding→LSTM→Flatten→Linear | text_cls | 747,522 | 查表式嵌入、RNN 串行递推 |
| `05-text-transformer.json` | TokenEmbedding→正弦位置编码→编码层→Flatten→Linear | text_cls | 1,420,674 | 自注意力、位置编码、与 04 对比并行性 |
| `06-mini-gpt-stack.json` | TokenEmbedding→RoPE→2×编码层→LMHead | random | 920,832 | decoder-only 栈的零件组合；无因果掩码的诚实边界 |
| `07-enc-dec-toy.json` | 编码器分支 + 解码器层（交叉注意力）+LMHead | random | 4,302,848 | enc-dec 三大家族、memory 端口 |
| `08-lora-parallel.json` | Linear ∥ LoRAAdapter → Add → Linear | random | 19,076 | LoRA 低秩旁路、up 零初始化、Add 搭旁路 |
| `09-qwen2.5-05b.json` | Qwen2.5-0.5B 复刻：TokenEmbedding→2×(RMSNorm→RoPE→自注意力→Add→RMSNorm→silu MLP→Add)→RMSNorm→LMHead | —（勿训练，读结构） | 296,147,584 | 现代 LLM 配方：pre-LN RMSNorm + RoPE + SiLU-MLP；description 字段列了与真实 Qwen 的 7 条差异 |

## 训练建议

- **mnist 组（01/02/03）**：训练对话框选 `mnist`，epochs 5、batch 64 就能看到清晰 loss 下降与准确率爬升。
- **text_cls 组（04/05）**：选 `text_cls`，seq 长度保持默认 32（04/05 的分类头按 seq=32 展平，改长度需同步改 Linear 的 in_features——这本身就是个练习）。
- **random 组（06/07/08）**：这些图输出的是全序列 logits 或多输入结构，不满足分类数据集契约，选 `random`（06 建议 num_samples≤256、epochs≤2）。它们的主要价值是「生成代码」后阅读结构。
- **09（Qwen 复刻）是纯结构阅读模型，不要训练**——顺便想想为什么：random 数据集会把 `[512, 64, 151936]` 的目标张量整个造出来（约 20 GB），而真实 LLM 的交叉熵只需要"下一个 token 的 id"，标签是 `[batch, seq]` 的整数矩阵。这就是 LLM 训练省内存的第一课。

## 读代码建议

导入后点「生成代码」，对照 `docs/大模型知识教程.md` 逐节阅读生成的 `model.py`：

- 01/02：找 `forward` 里的矩阵乘与展平；
- 03/08：找残差 `+` 与 LoRA 的 `down/up` 零初始化；
- 05/07：找 `MultiheadAttention` 调用与位置编码表；
- 06：找 `RotaryEmbedding`（RoPE 助手类会被注入到文件头部）。

## 改着玩（刻意练习）

1. 把 01 的 `Linear(128)` 中间层删掉或加宽，观察参数量与准确率变化；
2. 把 05 的 `dim_feedforward` 从 256 改成 128，看参数量省了多少（FFN 是参数大头）；
3. 在 02 里把第二个 MaxPool 删掉，观察校验器如何拒绝形状不匹配（28×28 展平后对不上 1568）；
4. 把 08 的 `rank` 改成 64，对比 08 的参数量变化（12,288 → 98,304）。
5. 导入 09 点「生成代码」，对照 `docs/大模型知识教程.md` 第 3/5 节，在生成的 `model.py` 里找出与真实 Qwen 的差异点：RMSNorm 的位置（pre-LN）、RoPE 挂载点（真实在注意力内部旋转 Q/K）、以及描述字段里列的 GQA/因果掩码缺失。
