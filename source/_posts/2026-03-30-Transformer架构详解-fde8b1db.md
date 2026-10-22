---
title: Transformer架构详解
date: 2026-03-30 00:00:56
categories: [教程]
tags: ["自动化", "技术", "学习"]
cover: https://images.unsplash.com/photo-7857?auto=format&fit=crop&w=1200&h=630&q=80&crop=entropy
---

# Transformer架构详解：从注意力机制到现代应用

## 引言

在深度学习领域，序列建模一直是一个核心挑战。传统的循环神经网络（RNN）和长短期记忆网络（LSTM）在处理长序列时面临着梯度消失和并行化困难的问题。2017年，Google的研究团队在论文《Attention Is All You Need》中提出了Transformer架构，彻底改变了自然语言处理（NLP）领域的格局。

Transformer的核心创新在于完全摒弃了循环结构，转而使用**自注意力机制**（Self-Attention）来捕捉序列中的依赖关系。这一设计不仅解决了长距离依赖问题，还实现了高度的并行化，使得模型训练效率大幅提升。如今，Transformer已成为BERT、GPT、T5等现代NLP模型的基石，并扩展到计算机视觉、语音处理等多个领域。

## 技术原理详解

### 1. 核心组件：自注意力机制

自注意力机制允许序列中的每个位置直接与其他所有位置交互，计算它们之间的相关性权重。

#### 注意力公式

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: 查询矩阵 [batch_size, seq_len, d_k]
    K: 键矩阵 [batch_size, seq_len, d_k]
    V: 值矩阵 [batch_size, seq_len, d_v]
    mask: 可选掩码矩阵
    """
    d_k = Q.size(-1)
    
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用掩码（如需要）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 应用softmax获取注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

**技术术语解释**：
- **查询（Query）**：当前关注的位置
- **键（Key）**：被比较的位置
- **值（Value）**：实际要聚合的信息
- **缩放因子**：√d_k，防止softmax梯度消失

### 2. 多头注意力

多头注意力允许模型同时关注不同表示子空间的信息：

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        """将输入分割为多个头"""
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换并分割多头
        Q = self.split_heads(self.W_q(Q), batch_size)
        K = self.split_heads(self.W_k(K), batch_size)
        V = self.split_heads(self.W_v(V), batch_size)
        
        # 计算缩放点积注意力
        attention_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 输出线性变换
        output = self.W_o(attention_output)
        
        return output
```

### 3. Transformer编码器层

每个编码器层包含多头注意力和前馈网络：

```python
class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 多头自注意力子层
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 4. 位置编码

由于Transformer没有循环结构，需要显式地添加位置信息：

```python
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

## 实战代码示例

### 完整Transformer编码器实现

```python
class TransformerEncoder(torch.nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, 
                 num_layers, max_len, dropout=0.1):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(d_model)
        
    def forward(self, src, src_mask=None):
        # 嵌入和位置编码
        x = self.embedding(src)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, src_mask)
            
        return self.norm(x)

# 使用示例
encoder = TransformerEncoder(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    max_len=100
)

# 模拟输入
batch_size = 32
seq_len = 50
src = torch.randint(0, 10000, (batch_size, seq_len))

# 前向传播
output = encoder(src)
print(f"输入形状: {src.shape}")
print(f"输出形状: {output.shape}")
```

### 文本分类任务示例

```python
class TransformerClassifier(torch.nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, 
                 num_layers, max_len, num_classes, dropout=0.1):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads, d_ff, 
            num_layers, max_len, dropout
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x, mask=None):
        # 获取序列表示
        encoded = self.encoder(x, mask)
        
        # 使用[CLS]标记进行分类
        cls_representation = encoded[:, 0, :]
        
        # 分类
        logits = self.classifier(cls_representation)
        
        return logits

# 训练示例
model = TransformerClassifier(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=4