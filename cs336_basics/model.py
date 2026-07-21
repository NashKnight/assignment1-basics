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
        x: torch.Tensor (batch, seq, d_in)
        """
        return x @ self.weight.T  # (batch, seq, d_out)

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
        token_ids: torch.Tensor (batch, seq)
        """
        # 取出 self.weight 中 token_ids 的值对应的那行就是该 id 的 embedding
        return self.weight[token_ids]  # (batch, seq, d_model)

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
        x: torch.Tensor (batch, seq, d_model)
        """
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)  # (batch, seq, 1)
        result = x / rms * self.weight  # (batch, seq, d_model)
        return result.to(in_type)

def silu(x):
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
        x: torch.Tensor (batch, seq, d_model)
        """
        return self.w2(silu(self.w1(x)) * self.w3(x))