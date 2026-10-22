---
title: Transformer架构详解
date: 2026-05-02 00:00:29
categories: [编程]
tags: ["学习", "教程", "自动化"]
cover: https://images.unsplash.com/photo-4168?auto=format&fit=crop&w=1200&h=630&q=80&crop=entropy
---

# Transformer架构详解：从原理到实战

## 一、引言：为什么需要Transformer？

在Transformer出现之前，序列到序列（Seq2Seq）任务主要依赖循环神经网络（RNN）或长短期记忆网络（LSTM）。然而，这些模型存在两个核心痛点：
- **并行计算困难**：RNN必须按时间步顺序计算，无法充分利用GPU并行能力
- **长距离依赖问题**：即使LSTM通过门控机制缓解了梯度消失，但当序列长度超过100时，信息依然容易丢失

2017年，Google团队在论文《Attention Is All You Need》中提出了Transformer架构，完全摒弃了循环结构，仅依赖注意力机制（Attention Mechanism），一举解决了上述问题。如今，BERT、GPT系列、T5等主流大模型均基于Transformer构建。

---

## 二、技术原理详解

### 2.1 整体架构

Transformer采用编码器-解码器（Encoder-Decoder）结构。以机器翻译为例：编码器读取源语言句子，解码器逐词生成目标语言。

```
输入序列 → [编码器 × N] → 编码表示 → [解码器 × N] → 输出序列
```

每个编码器层包含两个子层：多头自注意力（Multi-Head Self-Attention）和前馈神经网络（Feed-Forward Network）。解码器在此基础上增加了一个交叉注意力（Cross-Attention）子层，用于关注编码器的输出。

### 2.2 核心机制：自注意力（Self-Attention）

自注意力允许模型在计算某个位置的表示时，动态地关注序列中所有其他位置。其计算过程分为三步：

**步骤1：生成Query、Key、Value**
对每个输入向量x，通过三个不同的权重矩阵Wq、Wk、Wv生成Q、K、V：
```
Q = x · Wq
K = x · Wk
V = x · Wv
```

**步骤2：计算注意力分数**
使用Q和K计算相似度，公式如下（d_k为Key的维度，用于缩放防止梯度消失）：
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

**步骤3：加权求和**
将softmax后的权重与V相乘，得到每个位置的输出。权重越大的位置，对当前词的贡献越大。

### 2.3 多头注意力（Multi-Head Attention）

单一注意力可能只捕捉一种关系模式。多头注意力将Q、K、V分别拆分为h个头，每个头独立计算注意力，最后拼接并线性变换：
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
```
其中每个head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

例如，在翻译句子“The animal didn’t cross the street because it was too tired”时，不同的头可能分别关注“it”与“animal”的语法关系、与“tired”的语义关联。

### 2.4 位置编码（Positional Encoding）

由于Transformer没有循环结构，必须显式注入位置信息。原论文使用正弦/余弦函数：
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
其中pos为位置索引，i为维度索引。这种编码使模型能够通过相对位置学习距离关系。

---

## 三、实战代码示例

### 3.1 使用PyTorch实现自注意力层

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        Q = self.W_q(x)  # [batch, seq_len, d_k]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

# 使用示例
batch_size, seq_len, d_model = 2, 10, 512
x = torch.randn(batch_size, seq_len, d_model)
attention = SelfAttention()
output, weights = attention(x)
print(f"输出形状: {output.shape}")  # [2, 10, 64]
print(f"注意力权重形状: {weights.shape}")  # [2, 10, 10]
```

### 3.2 使用Hugging Face加载预训练BERT

```python
from transformers import BertTokenizer, BertModel
import torch

# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 编码输入文本
text = "Transformer architecture changed NLP forever."
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# 前向传播
with torch.no_grad():
    outputs = model(**inputs)
    
# 获取编码表示
last_hidden_state = outputs.last_hidden_state  # [1, seq_len, 768]
pooler_output = outputs.pooler_output          # [1, 768] 用于分类任务
print(f"序列表示形状: {last_hidden_state.shape}")
print(f"池化输出形状: {pooler_output.shape}")
```

### 3.3 自定义一个迷你Transformer编码器层

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        # 层归一化和dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 子层1：多头自注意力 + 残差连接
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 子层2：前馈网络 + 残差连接
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

# 测试
layer = TransformerEncoderLayer(d_model=512, n_heads=8)
x = torch.randn(10, 2, 512)  # [seq_len, batch, d_model]
output = layer(x)
print(f"编码器层输出形状: {output.shape}")  # [10, 2, 512]
```

---

## 四、最佳实践建议

### 4.1 训练技巧

1. **学习率预热（Learning Rate Warmup）**：Transformer对初始学习率敏感。建议前几个step（如4000步）线性增加学习率，之后按倒数平方根衰减。

2. **梯度裁剪（Gradient Clipping）**：设置max_grad_norm=1.0，防止梯度爆炸。

3. **混合精度训练**：使用torch.cuda.amp自动混精度训练，可提速30%-50%且几乎不损失精度。

### 4.2 架构选择

| 场景 | 推荐模型 | 说明 |
|------|----------|------|
| 文本分类/序列标注 | BERT-base | 编码器架构，双向上下文 |
| 文本生成（对话/故事） | GPT-2/GPT-3 | 解码器架构，自回归生成 |
| 翻译/摘要 | T5-base | 编码器-解码器架构 |
| 长文本处理（>512 tokens） | Longformer/BigBird | 使用稀疏注意力降低复杂度 |

### 4.3 性能优化

- **注意力计算优化**：对于长序列（>1024），使用FlashAttention或稀疏注意力替代标准注意力
- **模型并行**：当模型过大无法单卡加载时，使用张量并行（Tensor Parallelism）将参数分到多卡
- **推理加速