import torch

def cross_entropy(logits, targets):
    """
    logits: torch.Tensor (..., vocab_size)
    targets: torch.Tensor (..., )
    """
    log_z = torch.logsumexp(logits, dim=-1)  # (..., )
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (..., ), 这里 gather 要求 index 和 input 的维数一致，所以要先 unsqueeze 后再 squeeze
    return torch.mean(log_z - target_logits)

def gradient_clipping(params, max_l2_norm, eps=1e-6):
    """
    parameters: Iterable[torch.nn.Parameter]
    max_l2_norm: int
    eps: float
    """
    grads = [p.grad for p in params if p.grad is not None]  # 排除冻结/无梯度参数
    if not grads:
        return
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1:  # 缩放系数小于 1 才梯度裁剪，否则会放大
        for grad in grads:
            grad *= clip_coef
    return