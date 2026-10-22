---
title: Transformer架构详解
date: 2026-02-03 00:00:54
categories: [AI]
tags: ["Python", "教程", "编程"]
cover: https://images.unsplash.com/photo-5256?auto=format&fit=crop&w=1200&h=630&q=80&crop=entropy
---

# Transformer架构详解：从自注意力到现代应用

## 引言

在深度学习的发展历程中，序列建模一直是一个核心挑战。传统的循环神经网络（RNN）和长短期记忆网络（LSTM）在处理序列数据时面临着梯度消失、难以并行化等固有局限。2017年，Vaswani等人在论文《Attention Is All You Need》中提出的Transformer架构，彻底改变了自然语言处理领域的格局。

Transformer的核心创新在于完全摒弃了循环结构，转而使用自注意力机制（Self-Attention）来捕捉序列中的依赖关系。这一设计不仅解决了RNN的并行化问题，还显著提升了模型对长距离依赖的建模能力。如今，Transformer已成为BERT、GPT、T5等现代NLP模型的基石，并扩展到计算机视觉、语音处理等多个领域。

## 技术原理详解

### 1. 自注意力机制

自注意力机制是Transformer的核心组件，它允许序列中的每个位置直接与其他所有位置交互，从而计算每个位置的表示。

**技术术语解释**：
- **查询（Query）、键（Key）、值（Value）**：注意力机制中的三个核心概念，分别用于计算注意力权重和加权求和
- **缩放点积注意力（Scaled Dot-Product Attention）**：通过点积计算相似度，并用缩放因子防止梯度消失
- **多头注意力（Multi-Head Attention）**：并行运行多个注意力头，捕捉不同类型的依赖关系

自注意力的数学公式如下：

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

其中d_k是键向量的维度，缩放因子√d_k用于防止点积过大导致softmax梯度消失。

### 2. Transformer架构组成

完整的Transformer架构包含以下关键组件：

#### 2.1 编码器-解码器结构
- **编码器**：由N个相同的层堆叠而成，每层包含多头自注意力层和前馈神经网络
- **解码器**：同样由N个相同的层堆叠，但额外包含编码器-解码器注意力层

#### 2.2 位置编码
由于Transformer没有循环结构，需要显式地注入位置信息：
```python
import numpy as np

def positional_encoding(seq_len, d_model):
    """生成位置编码"""
    pos_enc = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return pos_enc
```

#### 2.3 前馈神经网络
每个注意力层后接一个前馈网络：
```
FFN(x) = max(0, xW1 + b1)W2 + b2
```

### 3. 关键技术细节

#### 3.1 残差连接和层归一化
每个子层后都应用残差连接和层归一化：
```
LayerNorm(x + Sublayer(x))
```

#### 3.2 掩码机制
- **填充掩码**：忽略填充位置
- **前瞻掩码**：防止解码器看到未来信息

## 实战代码示例

### 示例1：实现缩放点积注意力

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力实现"""
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        # query, key, value的形状: (batch_size, seq_len, d_model)
        d_k = query.size(-1)
        
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

# 使用示例
batch_size = 2
seq_len = 10
d_model = 64

attention = ScaledDotProductAttention()
query = torch.randn(batch_size, seq_len, d_model)
key = torch.randn(batch_size, seq_len, d_model)
value = torch.randn(batch_size, seq_len, d_model)

output, weights = attention(query, key, value)
print(f"输出形状: {output.shape}")  # [2, 10, 64]
print(f"注意力权重形状: {weights.shape}")  # [2, 10, 10]
```

### 示例2：多头注意力实现

```python
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def split_heads(self, x):
        """将输入分割为多个头"""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """合并多个头的输出"""
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        # 保存残差连接
        residual = query
        
        # 线性变换并分割头
        query = self.split_heads(self.W_q(query))
        key = self.split_heads(self.W_k(key))
        value = self.split_heads(self.W_v(value))
        
        # 应用注意力
        if mask is not None:
            mask = mask.unsqueeze(1)  # 为多头扩展维度
        
        attention_output, _ = self.attention(query, key, value, mask)
        
        # 合并头并线性变换
        attention_output = self.combine_heads(attention_output)
        output = self.W_o(attention_output)
        
        # 应用dropout、残差连接和层归一化
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        
        return output

# 测试多头注意力
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(4, 20, 512)  # batch_size=4, seq_len=20, d_model=512
output = mha(x, x, x)
print(f"多头注意力输出形状: {output.shape}")  # [4, 20, 512]
```

### 示例3：Transformer编码器层

```python
class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # 自注意力层
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力子层
        attention_output = self.self_attention(x, x, x, mask)
        
        # 前馈网络子层
        ff_output = self.feed_forward(attention_output)
        ff_output =