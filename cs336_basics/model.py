import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        in_features: int
        out_features: int
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        """
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        self.weight = nn.Parameter(torch.empty(self.d_out, self.d_in, device=device, dtype=dtype))  # (d_out, d_in)

        nn.init.trunc_normal_(self.weight, mean=0.0, std=math.sqrt(2 / (self.d_in + self.d_out)))

    def forward(self, x):
        """
        x: torch.Tensor (..., d_in)
        """
        return x @ self.weight.T  # (..., d_out)

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        num_embeddings: int
        embedding_dim: int
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        """
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.d_model, device=device, dtype=dtype))  # (vocab_size, d_model)

        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, token_ids):
        """
        token_ids: torch.Tensor (...)
        """
        # 取出 self.weight 中 token_ids 的值对应的那行就是该 id 的 embedding
        return self.weight[token_ids]  # (..., d_model)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        """
        d_model: int
        eps: float
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.d_model, device=device, dtype=dtype))  # (d_model,)

    def forward(self, x):
        """
        x: torch.Tensor (..., d_model)
        """
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)  # (..., 1)
        result = x / rms * self.weight  # (..., d_model)
        return result.to(in_type)

def silu(x):
    """
    x: torch.Tensor (...)
    """
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        """
        d_model: int
        d_ff: int
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # (d_ff, d_model)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # (d_model, d_ff)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # (d_ff, d_model)

    def forward(self, x):
        """
        x: torch.Tensor (..., d_model)
        """
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        """
        theta: float
        d_k: int
        max_seq_len: int
        device: torch.device | None = None
        """
        super().__init__()
        assert d_k % 2 == 0
        
        # 计算旋转角 angles，分母 freqs (freqs 取了倒数)，分子 positions
        freqs = theta ** (-torch.arange(0, d_k , 2, device=device).float() / d_k)  # (d_k // 2, )
        positions = torch.arange(max_seq_len, device=device).float().unsqueeze(1)  # (max_seq_len, 1)，unsqueeze 后可以和 freqs 利用广播相乘
        angles = positions * freqs  # (max_seq_len, d_k // 2)

        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)  # (max_seq_len, d_k // 2)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)  # (max_seq_len, d_k // 2)

    def forward(self, x, token_positions):
        """
        x: torch.Tensor (..., seq, d_k)
        token_positions: torch.Tensor (..., seq)
        """
        # cos 和 sin 从 buffer 中取，精度保持跟 x 一致
        cos = self.cos_cached[token_positions].to(x.dtype)  # (..., seq, d_k // 2)
        sin = self.sin_cached[token_positions].to(x.dtype)  # (..., seq, d_k // 2)
        # 拆分奇偶位置
        x_even = x[..., 0::2]  # (..., seq, d_k // 2)
        x_odd = x[..., 1::2]  # (..., seq, d_k // 2)
        # 奇偶位置分别应用旋转编码公式计算
        out_even = cos * x_even - sin * x_odd
        out_odd = sin * x_even + cos * x_odd
        # 输出合并奇偶位置
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd

        return out

def softmax(x, dim):
    """
    x: torch.Tensor (...)
    dim: int
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values  # dim 维为 1,其余维度不变
    x_exp = torch.exp(x - x_max)  # (...)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)  # (...)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: torch.Tensor (..., seq_q, d_k)
    K: torch.Tensor (..., seq_k, d_k)
    V: torch.Tensor (..., seq_k, d_v)
    mask: torch.Tensor (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (..., seq_q, seq_k)
    if mask is not None:  # 注意整个课程中定义 mask=True 为要保留的位置， False 是需要 mask 的位置
        scores = scores.masked_fill(~mask, float("-inf"))  # (..., seq_q, seq_k)
    attn_weights = softmax(scores, dim=-1)  # (..., seq_q, seq_k)
    return attn_weights @ V  # (..., seq_q, d_v)

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None, device=None, dtype=None):
        """
        d_model: int
        num_heads: int
        theta: float
        max_seq_len: int
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        """
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k, self.d_v = d_model // num_heads, d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)  # (d_model, d_model)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)  # (d_model, d_model)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)  # (d_model, d_model)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)  # (d_model, d_model)

        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x, token_positions=None):
        """
        x: torch.Tensor (..., seq, d_model)
        token_positions: torch.Tensor (..., seq)
        """
        # 获取 seq 维度
        *leading, seq, _ = x.shape

        # 计算 QKV 矩阵
        Q = self.q_proj(x)  # (..., seq, d_model)
        K = self.k_proj(x)  # (..., seq, d_model)
        V = self.v_proj(x)  # (..., seq, d_model)
        
        # 分头
        Q = Q.view(*leading, seq, self.num_heads, self.d_k).transpose(-3, -2)  # (..., num_heads, seq, d_k)
        K = K.view(*leading, seq, self.num_heads, self.d_k).transpose(-3, -2)  # (..., num_heads, seq, d_k)
        V = V.view(*leading, seq, self.num_heads, self.d_v).transpose(-3, -2)  # (..., num_heads, seq, d_v)

        # 加入旋转位置编码
        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # 构造 causal mask，用 tril 构造下三角矩阵，默认 diagonal=0 对角线为 True
        causal_mask = torch.tril(torch.ones(seq, seq, device=x.device, dtype=torch.bool))  # (seq, seq)

        # 计算点积自注意力
        out = scaled_dot_product_attention(Q, K, V, causal_mask)  # (..., num_heads, seq, d_v)
        out = out.transpose(-3, -2).contiguous().view(*leading, seq, self.d_model)  # (..., seq, d_model)
        return self.output_proj(out)  # (..., seq, d_model)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=None, max_seq_len=None, device=None, dtype=None):
        """
        d_model: int
        num_heads: int
        d_ff: int
        theta: float
        max_seq_len: int
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        """
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
    
    def forward(self, x, token_positions=None):
        """
        x: torch.Tensor (..., seq, d_model)
        token_positions: torch.Tensor (..., seq)
        """
        if token_positions is None:
            token_positions = torch.arange(x.size(-2), device=x.device)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, vocab_size, context_length, num_layers, theta=None, device=None, dtype=None):
        """
        d_model: int
        num_heads: int
        d_ff: int
        vocab_size: int
        context_length: int
        num_layers: int
        theta: float
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        """
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta=theta, max_seq_len=context_length, device=device, dtype=dtype) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        """
        x: torch.Tensor (..., seq)  # token ids
        token_positions: torch.Tensor (..., seq) | None
        returns: torch.Tensor (..., seq, vocab_size)
        """
        x = self.token_embeddings(x)  # (..., seq, d_model)
        for layer in self.layers:
            x = layer(x, token_positions)  # (..., seq, d_model)
        return self.lm_head(self.ln_final(x))  # (..., seq, vocab_size)