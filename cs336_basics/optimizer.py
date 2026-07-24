import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        # params 是要优化的模型参数，defaults 是自己注册的超参数
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # init: 1. params 组织成 self.param_groups；2. 每个 group 补上超参数；3. 建立空的 self.state
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # 取出超参数
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            # 权重参数更新
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 取出依赖参数的变量：时间步 t 和梯度 grad
                state = self.state[p]
                if len(state) == 0:  # 初始化 state
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                t, m, v = state["t"], state["m"], state["v"]
                grad = p.grad.data
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                # 衰减
                p.data -= lr * weight_decay * p.data
                # 计算一阶和二阶动量
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                state["m"], state["v"] = m, v
                # 权重梯度和时间步更新
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                state["t"] = t + 1

        return loss

def get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    """
    t: 当前时间步
    alpha_max: 最大学习率
    alpha_min: 最小学习率
    T_w: warm-up 迭代轮数
    T_c: 余弦退火的总迭代轮数
    return: alpha_t, 当前时间步的学习率
    """
    if t < T_w:  # warm-up
        alpha_t = t / T_w * alpha_max
    elif t < T_c:  # cosine annealing
        alpha_t = alpha_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
    else:  # post-annealing
        alpha_t = alpha_min
    return alpha_t