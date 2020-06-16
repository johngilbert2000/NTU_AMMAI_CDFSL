# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

import utils

class ProtoMarginTrain(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, m=0.50, Lambda=0.50):
        super(ProtoMarginTrain, self).__init__( model_func,  n_way, n_support)
        self.loss_fn  = nn.CrossEntropyLoss()
        self.Lambda = Lambda
        self.m = m


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )


        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        return scores

    def set_forward_margin_loss(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_proto     = F.normalize(z_proto)

        dists = euclidean_dist(z_proto, z_proto)
        dists = torch.sqrt(dists + 1e-9) # Add small value to avoid nan gradient
        dists = self.m - dists
        zeros = torch.zeros(size=dists.shape, dtype=torch.float32).cuda()
        dists = torch.max(dists, zeros)
        dists = torch.triu(dists, diagonal=1)

        loss = torch.sum(torch.pow(dists, 2)) / (self.n_way * (self.n_way-1) / 2)

        return loss

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)
        loss = self.loss_fn(scores, y_query)
        margin_loss = self.set_forward_margin_loss(x)

        total_loss = loss + self.Lambda * margin_loss
        return total_loss

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

# python ./train.py --dataset miniImageNet --model ResNet10  --method protomargin --n_shot 5 --train_aug