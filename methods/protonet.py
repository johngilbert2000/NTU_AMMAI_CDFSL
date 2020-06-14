# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

import utils


####
import torch
import torchvision

from torch import optim, nn
from torch.nn import *
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

# from PIL import Image
# from pathlib import Path

# for ROC curve
# from sklearn import metrics
# from scipy import interp

import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt #, mpld3
# import matplotlib

import math
from typing import *
import time
import datetime

# from IPython.display import display, clear_output
# from IPython.display import HTML


###
import warnings

# from torch.nn.module import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction

from torch import Tensor
from typing import Optional


class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
            

class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)

class CrossEntropyLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         print('input',input)
#         print('target',target)
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)




class LargeMarginCosineLoss(nn.Module):
    """
    Reference: 
    H. Wang et al. CosFace: Large Margin Cosine Loss for Deep Face Recognition
    https://arxiv.org/pdf/1801.09414.pdf
    
    Also referenced cvqluu's implementation of Angular Penalty Loss:
    https://paperswithcode.com/paper/cosface-large-margin-cosine-loss-for-deep
    """
    
    def __init__(self, in_features=5, out_features=2, s=64.0, m=0.35):
        super(LargeMarginCosineLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # cos(θ_j,i) = W_j^T * x_i
        self.linear = Linear(in_features, out_features, bias=False)
        
    def forward(self, x, targets):
        
        # normalize
        x = F.normalize(x, p=2, dim=1)
        for W in self.linear.parameters():
            W = F.normalize(W, p=2, dim=1)
            
        cos_θ = self.linear(x)
        s_cos_θ = self.s*torch.diagonal(cos_θ.transpose(0,1)[targets]-self.m)
#         print(s_cos_θ)
        try:
            cos_θj = [torch.cat((cos_θ[j,:y], cos_θ[j,(y+1):])).unsqueeze(0) for j, y in zip(len(targets), targets)] # <<<-- issue
            sum_j = torch.sum(torch.exp(self.s*torch.cat(cos_θj, dim=0)), dim=1)
        except:
            raise ValueError(cos_θ)
        
        result = torch.mean(torch.log(torch.exp(s_cos_θ) + sum_j) - torch.log(torch.exp(s_cos_θ)))
        
        return result



class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(ProtoNet, self).__init__( model_func,  n_way, n_support)
#         self.loss_fn  = nn.CrossEntropyLoss()
#         self.loss_fn = LargeMarginCosineLoss()
        self.loss_fn = CrossEntropyLoss()


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )


        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)
#         print('scores', scores.size())
        loss = self.loss_fn(scores, y_query)
        if isinstance(loss, float):
            pass
#             print('loss',loss)
        else:
            pass
#             print('> loss', loss)
        return loss

def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
