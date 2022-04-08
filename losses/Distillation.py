from torch import nn
import torch.nn.functional as F

class Distillation(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.T = temperature
        self.criterion = nn.KLDivLoss()

    def forward(self, z_s, z_t):

        return self.criterion(F.log_softmax(z_s / self.T), F.softmax(z_t / self.T)) # * (self.T * self.T)