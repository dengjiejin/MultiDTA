import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def loss_function(y, t, drop_rate):
    # loss = torch.nn.MSELoss(y, t, reduce = None)
    loss = torch.nn.functional.mse_loss(y, t, reduce=False)

    loss_mul = loss * t
    ind_sorted = np.argsort(loss_mul.cpu().data, axis=0).cuda()
    ind_sorted= ind_sorted.squeeze(1)
    loss_sorted = loss[ind_sorted]


    remember_rate = drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[num_remember:]

    loss_update = torch.nn.functional.mse_loss(y[ind_update], t[ind_update])
    # loss_update = torch.nn.MSELoss(y[ind_update], t[ind_update])

    return loss_update
