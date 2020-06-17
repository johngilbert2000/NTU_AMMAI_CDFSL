import backbone
import utils
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class ArcFaceTrain(nn.Module):
    def __init__(self, model_func, num_class, s=4.0, m=0.50, easy_margin=False, pretrain=False):
        super(ArcFaceTrain, self).__init__()
        self.feature    = model_func()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.pretrain = pretrain

        self.weight = nn.Parameter(torch.FloatTensor(num_class, self.feature.final_feat_dim))
        nn.init.xavier_uniform_(self.weight)
        self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
        self.classifier.bias.data.fill_(0)

        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()

    def forward(self,x):
        x    = Variable(x.cuda())
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def forward_feature(self,x):
        x     = Variable(x.cuda())
        feat  = self.feature.forward(x)
        return feat

    def forward_arcface_loss(self, x, y):
        y = Variable(y.cuda())

        feat = self.forward_feature(x)
        cosine = F.linear(F.normalize(feat + 1e-9), F.normalize(self.weight + 1e-9))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2) + 1e-9).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m * self.s + 1e-9
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, (cosine - self.mm) * self.s)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        scores = (one_hot * phi) + ((1.0 - one_hot) * cosine * self.s) + 1e-9

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))  

        return self.loss_fn(scores, y )

    def forward_loss(self, x, y):
        y = Variable(y.cuda())

        scores = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))  

        return self.loss_fn(scores, y )
    
    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0
        for i, (x,y) in enumerate(train_loader):
            if self.pretrain and epoch < 1:
                loss = self.forward_loss(x, y)
            else:
                loss = self.forward_arcface_loss(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), self.top1.val, self.top1.avg))
                     
    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration

class CosFaceTrain(nn.Module):
    def __init__(self, model_func, num_class, s=30.0, m=0.35, pretrain=False):
        super(CosFaceTrain, self).__init__()
        self.feature    = model_func()
        self.s = s
        self.m = m

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.pretrain = pretrain

        self.weight = nn.Parameter(torch.FloatTensor(num_class, self.feature.final_feat_dim))
        nn.init.xavier_uniform_(self.weight)
        self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
        self.classifier.bias.data.fill_(0)

        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()

    def forward(self,x):
        x    = Variable(x.cuda())
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def forward_feature(self,x):
        x     = Variable(x.cuda())
        feat  = self.feature.forward(x)
        return feat

    def forward_cosface_loss(self, x, y):
        y = Variable(y.cuda())

        feat = self.forward_feature(x)
        cosine = F.linear(F.normalize(feat), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        scores = (one_hot * phi * self.s) + ((1.0 - one_hot) * cosine * self.s)
        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))  

        return self.loss_fn(scores, y )

    def forward_loss(self, x, y):
        y = Variable(y.cuda())

        scores = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))  

        return self.loss_fn(scores, y )
    
    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0
        for i, (x,y) in enumerate(train_loader):
            if self.pretrain and epoch < 1:
                loss = self.forward_loss(x, y)
            else:
                loss = self.forward_cosface_loss(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), self.top1.val, self.top1.avg))
                     
    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration

class SphereFaceTrain(nn.Module):
    def __init__(self, model_func, num_class, m=4, pretrain=False):
        super(SphereFaceTrain, self).__init__()
        self.feature    = model_func()
        self.m = m
        self.pretrain = pretrain

        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = nn.Parameter(torch.FloatTensor(num_class, self.feature.final_feat_dim))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]
        self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
        self.classifier.bias.data.fill_(0)

        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()

    def forward(self,x):
        x    = Variable(x.cuda())
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def forward_feature(self,x):
        x     = Variable(x.cuda())
        feat  = self.feature.forward(x)
        return feat

    def forward_sphereface_loss(self, x, y):
        y = Variable(y.cuda())

        feat = self.forward_feature(x)

        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        cos_theta = F.linear(F.normalize(feat), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(feat, 2, 1)

        one_hot = torch.zeros(cos_theta.size(), device='cuda')
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)

        scores = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        scores *= NormOfFeature.view(-1, 1)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))  

        return self.loss_fn(scores, y )

    def forward_loss(self, x, y):
        y = Variable(y.cuda())

        scores = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))  

        return self.loss_fn(scores, y )
    
    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0
        for i, (x,y) in enumerate(train_loader):
            if self.pretrain and epoch < 1:
                loss = self.forward_loss(x, y)
            else:
                loss = self.forward_sphereface_loss(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), self.top1.val, self.top1.avg))
                     
    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration

# python ./train.py --dataset miniImageNet --model ResNet10  --method ArcFace --train_aug
# python ./train.py --dataset miniImageNet --model ResNet10  --method ArcFace-pretrain --train_aug
# python ./train.py --dataset miniImageNet --model ResNet10  --method CosFace --train_aug
# python ./train.py --dataset miniImageNet --model ResNet10  --method CosFace-pretrain --train_aug
# python ./train.py --dataset miniImageNet --model ResNet10  --method SphereFace --train_aug
# python ./train.py --dataset miniImageNet --model ResNet10  --method SphereFace-pretrain --train_aug
