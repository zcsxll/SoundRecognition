import torch

def clip_grad_norm(parameters, max_norm, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type <= 0:
        total_norm = max(p.grad.abs().max() for p in parameters)
    else:
        total_norm = 0.0
        for p in parameters:
            total_norm += p.grad.norm(norm_type).item() ** norm_type
        total_norm **= (1.0 / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            p.grad.mul_(clip_coef)
    return total_norm