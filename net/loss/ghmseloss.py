import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

class GHM_Loss(nn.Module):
    def __init__(self, bins, alpha):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        a = (g - g.min()) / (g.max() - g.min())
        return torch.floor(a * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.000001)
        beta = N / gd
        return self._custom_loss(x, target, beta[bin_idx])

class GHMSE_Loss(GHM_Loss):
    def __init__(self, bins, alpha, mu):
        super(GHMSE_Loss, self).__init__(bins, alpha)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        d = x - target
        loss = d.pow(2)
        N = 2 * x.size(0) * x.size(1)
        return (loss * weight.cuda()).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        N = x.size(0) * x.size(1)
        return d / N