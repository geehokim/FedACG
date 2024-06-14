import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helper import *

__all__ = ['FedLC', 'FedDecorrLoss','KD']


def FedLC(label_distrib, logit, y, tau):
    cal_logit = torch.exp(logit- (tau* torch.pow(label_distrib, -1 / 4).unsqueeze(0).expand((logit.shape[0], -1))))
    #breakpoint()
    y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
    sum_y_logit = cal_logit.sum(dim=-1, keepdim=True)
    #loss = -torch.log(y_logit / (sum_y_logit - y_logit))
    loss = -torch.log(y_logit / (sum_y_logit))
    return loss.sum() / logit.shape[0]

class FedDecorrLoss(nn.Module):

    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        N, C = x.shape
        if N == 1:
            return 0.0

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N

        return loss


def KD(input_p, input_q, T=1):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    p = F.softmax(input_p/T, dim = 1)
    q = F.log_softmax(input_q/T, dim = 1)
    result = kl_loss(q,p)
    return result    