import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


#### Cells
# convolution cell
class conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.cell=nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self,x):
        return self.cell(x)
    
# dense cell: 
class dense(nn.Module):
    def __init__(self,ch_in=512,ch_out=512,squeeze_input=False):
        super().__init__()
        self.cell=nn.Sequential(
            nn.Linear(ch_in, ch_out),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(),
        )
        self.squeeze_input=squeeze_input
    def forward(self,x):
        if self.squeeze_input:
            return self.cell(x.squeeze())
        return self.cell(x)
    
#### Backbone
class FeatureExtractor(nn.Module):
    def __init__(self,
                 in_features=3,
                 latent_features=3,
                 hidden_featrues=(64,128,256)):
        super().__init__()
        self.blocks = nn.Sequential(
            conv(3,hidden_featrues[0]),
            *[conv(ch1, ch2) for ch1,ch2 in zip(hidden_featrues[:-1],hidden_featrues[1:])],
            nn.AdaptiveAvgPool2d((1,1)),
            dense(hidden_featrues[-1],latent_features,squeeze_input=True)
        )
    def forward(self, x):
        return self.blocks(x)
    
#### Head

def cosine(x,w):
    return F.linear(F.normalize(x,dim=1), F.normalize(w,dim=1))

def euc_dist(x,w):
    return F.pairwise_distance(x, w, 2)
class MetricLayer(nn.Module):
    def __init__(self, n_in_features,n_out_features=10,metric=cosine):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_out_features, n_in_features))
        nn.init.xavier_uniform_(self.weight,gain=1.0)
        self.metric=metric
    def forward(self,x):
        return self.metric(x,self.weight)
    
#### Model
class ClasModel(nn.Module):
    def __init__(self,ways,shots,backbone,head,metric=cosine,dropout=False,fp16=False,using_insightface=False):
        super().__init__()
        self.ways=ways
        self.shots=shots
        self.fp16=fp16
        self.using_insightface=using_insightface
        self.backbone=backbone
        self.head=head
        self.metric=metric
        self.dropout=dropout
        
        neck_layers=[]
        if using_insightface:
            neck_layers.append(nn.Dropout(0.2))
            neck_layers.append(nn.AdaptiveAvgPool2d((1,1)))
            neck_layers.append(nn.Flatten(1))
            neck_layers.append(nn.Linear(512, self.backbone.num_features))
            if dropout:
                neck_layers.append(nn.Dropout(0.2))
        elif dropout:
            neck_layers.append(nn.Dropout(0.2))
            
        self.neck=nn.Sequential(*neck_layers)
        
        
        
class Baseline(ClasModel):
    def __init__(self,ways,backbone,head,metric=cosine,dropout=False,fp16=False,using_insightface=False):
        assert(backbone is not None)
        super().__init__(ways,None,backbone,head,metric,dropout,fp16,using_insightface)
        
    def forward(self,data,label=None):
        
        # Transfer Learing: backbone+ output head
        latent=self.backbone(data)
        if self.neck:
            latent=self.neck(latent)
        logits=self.head(latent)
        return logits

class SiameseNet(ClasModel):
    def __init__(self,backbone,metric=cosine,dropout=False,fp16=False,using_insightface=False):
        super().__init__(2,None,backbone,None,metric,dropout,fp16,using_insightface)

    def forward(self,data,label=None):
        # ???Embedding
        latent=[*map(self.backbone,data.transpose(0,1))]    
        if self.neck:
            latent=[*map(self.neck,latent)]
        # latent???metric ?????????cosine
        logits=torch.stack([*map(self.metric,latent[0],latent[1])],dim=0)
        return logits

class PrototypicalNet(ClasModel):
    def __init__(self,ways,shots,backbone,metric=cosine,dropout=False,fp16=False,using_insightface=False):
        super().__init__(ways,shots,backbone,None,metric,dropout,fp16,using_insightface)
        
    def meta_forward(self,dataset,label=None):
        # ?????????embedding
        latent=torch.stack([*map(self.backbone,dataset)],dim=0)
        if self.neck:
            latent=[*map(self.neck,latent)]
        
        latent_q,latent_s=latent[:,self.ways*self.shots:],latent[:,:self.ways*self.shots]
        # ?????? prototypes
        latent_proto=torch.stack([torch.mean(l,dim=1) for l in torch.split(latent_s,self.shots,dim=1)],dim=1)
        logits=torch.cat([*map(self.metric,latent_q,latent_proto)],dim=0)
        return logits
    def save_prototypes(self,dataset):
        with torch.no_grad():
            if self.neck:
                x=self.neck(dataset)
            else:
                x=dataset
            latent_s=self.backbone(x)
            
        self.weight=nn.Parameter(
            torch.stack(
                [torch.mean(l,dim=0) for l in torch.split(latent_s,self.shots,dim=0)]
            ,dim=0)
        )
    def forward(self,data,label=None):
        if self.neck:
            x=self.neck(data)
        else:
            x=data
        latent_q=self.backbone(x)
        logits=self.metric(latent_q,self.weight)
        return logits
#### Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-10,ce=nn.CrossEntropyLoss()):
        super().__init__()
        self.gamma = gamma
        self.eps = torch.tensor(eps,dtype=torch.float32)
        self.ce = ce
    def forward(self,  y_pred,y_true):
        # ??????cross entropy
        L=self.ce(y_pred+self.eps, y_true)
        # ????????????gamma????????????entropy????????????(???????????????)
        p = torch.exp(-L)
        loss = (1 - p) ** self.gamma * L
        return loss.mean()
    
class ContrastiveLoss(nn.Module):
    def __init__(self,m=1):
        super().__init__()
        self.m=m
        self.activation=torch.sigmoid # ???cosine similarity???????????????output contrast
        self.loss_fn=nn.BCELoss() # ???cosine similarity???????????????output contrast
        self.z=torch.tensor(0.,dtype=torch.float32,requires_grad=False)
    def forward(self, y_pred,y_true):
        # ?????????????????????square
        # ????????????????????????margin- distance?????????distance??????margin????????????????????????distance
        loss=torch.mean(y_true * torch.square(y_pred)+ 
                        (1 - y_true)* torch.square(torch.maximum(self.m - y_pred, self.z)
                        ),dim=-1,keepdims=True)
        return loss.mean()
        
class AddMarginLoss(nn.Module):
    def __init__(self, s=15.0, m=0.40,ways=10,loss_fn=FocalLoss()):
        super().__init__()
        self.s = s
        self.m = m
        self.ways=ways
        self.loss_fn=loss_fn
        
    def forward(self, cosine, label=None):
        # ?????????cosine???margin
        cos_phi = cosine - self.m
        # ???onehot???????????????????????????margin???onehot???????????????margin     
        one_hot=F.one_hot(label, num_classes=self.ways).to(torch.float32)
        metric = (one_hot * cos_phi) + ((1.0 - one_hot) * cosine)
        # ?????????????????????
        metric *= self.s
        return self.loss_fn(metric,label)

class ArcMarginLoss(AddMarginLoss):
    def __init__(self, s=32.0, m=0.40,ways=10, easy_margin=False,loss_fn=FocalLoss()):
        ## ??????AddMarginLoss???????????????????????????
        super().__init__(s,m,ways,loss_fn)
        
        ## ??????????????????easy margin
        self.easy_margin = easy_margin
        ## ????????????arc margin?????????cosine??????sine ???
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # phi ???[0??,180??]??????????????????cos(phi+m)????????????
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        # ????????????0?????????????????????????????????
        self.eps = 1e-6
    def forward(self, cosine, label=None):
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + self.eps)
        # cos(phi)cos(m)-sin(phi)sin(m)??????cos(phi + m)
        # ??????margin?????????????????????phi????????????????????????????????????softmax(cos(phi))?????????
        cos_phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # cosine?????????????????????????????????phi margin
            cos_phi = torch.where(cosine > 0, cos_phi, cosine)
        else:
            # ??????????????????cosine(phi)??????margin?????????phi margin??????
            #          ???cosine(phi)??????margin?????????cosine margin??????
            cos_phi = torch.where(cosine > self.th, cos_phi, cosine - self.mm)
            
        # ???onehot???????????????????????????margin???onehot???????????????margin    
        one_hot=F.one_hot(label, num_classes=self.ways).to(torch.float32)
        metric = (one_hot * cos_phi) + ((1.0 - one_hot) * cosine)
        # ?????????????????????
        metric *= self.s
        return self.loss_fn(metric,label)