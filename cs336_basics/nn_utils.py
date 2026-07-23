import torch

def cross_entropy(logits, targets):
    """
    logits: torch.Tensor (..., vocab_size)
    targets: torch.Tensor (..., )
    """
    log_z = torch.logsumexp(logits, dim=-1)  # (..., )
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (..., ), 这里 gather 要求 index 和 input 的维数一致，所以要先 unsqueeze 后再 squeeze
    return torch.mean(log_z - target_logits)