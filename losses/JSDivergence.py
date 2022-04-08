import torch
from torch import nn
import torch.nn.functional as F

class JSDivergence(nn.Module):
    def __init__(self, lambd=12):
        super().__init__()
        self.name = "JSDivergence"
        self.lambd = lambd

    def forward(self, p0, pi_list, y=None):
        p_count = 1 + len(pi_list)
        p_mixture = p0.detach().clone()
        for pi in pi_list:
            p_mixture += pi
        p_mixture = torch.clamp(p_mixture / p_count, 1e-7, 1).log()

        loss = F.kl_div(p_mixture, p0, reduction='batchmean')
        for pi in pi_list:
            loss += F.kl_div(p_mixture, pi, reduction='batchmean')
        loss /= p_count

        return  self.lambd * loss