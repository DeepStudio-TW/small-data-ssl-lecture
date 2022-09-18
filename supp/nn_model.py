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
    return F.linear(F.normalize(x,dim=-1), F.normalize(w,dim=-1))
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
    def __init__(self,ways,shots,backbone,head,metric=cosine):
        super().__init__()
        self.ways=ways
        self.shots=shots

        self.backbone=backbone
        self.head=head
        self.metric=metric
        
class Baseline(ClasModel):
    def __init__(self,ways,backbone,head):
        assert(backbone is not None)
        super().__init__(ways,None,backbone,head)
    def forward(self,data,label=None):
        # Transfer Learing: backbone+ output head
        hidden=self.backbone(data)
        logits=self.head(hidden)
        return logits

class SiameseNet(ClasModel):
    def __init__(self,backbone,metric=cosine):
        super().__init__(2,None,backbone,None,metric)
    def forward(self,data,label=None):
        # 進Embedding
        latent=[*map(self.backbone,data.transpose(0,1))]
        # latent算metric 這邊用cosine
        logits=torch.stack([*map(self.metric,latent[0],latent[1])],dim=0)
        return logits

class PrototypicalNet(ClasModel):
    def __init__(self,ways,shots,backbone,metric=cosine):
        super().__init__(ways,shots,backbone,None,metric)
    def meta_forward(self,dataset,label=None):
        # 全部進embedding
        latent=torch.stack([*map(self.backbone,dataset)],dim=0)
        latent_q,latent_s=latent[:,self.ways*self.shots:],latent[:,:self.ways*self.shots]
        # 計算 prototypes
        latent_proto=torch.stack([torch.mean(l,dim=1) for l in torch.split(latent_s,self.shots,dim=1)],dim=1)
        logits=torch.cat([*map(self.metric,latent_q,latent_proto)],dim=0)
        return logits
    def save_prototypes(self,dataset):
        with torch.no_grad():
            latent_s=self.backbone(dataset)
        self.weight=nn.Parameter(
            torch.stack(
                [torch.mean(l,dim=0) for l in torch.split(latent_s,self.shots,dim=0)]
            ,dim=0)
            )
    def forward(self,data,label=None):
        latent_q=self.backbone(data)
        logits=self.metric(latent_q,self.weight)
        return logits
#### Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-10):
        super().__init__()
        self.gamma = gamma
        self.eps = torch.tensor(eps,dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss()
    def forward(self,  y_pred,y_true):
        # 計算cross entropy
        logp = self.ce(y_pred+self.eps, y_true)
        # 計算乘上gamma次方後的entropy反方機率(將對比放大)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
    
class ContrastiveLoss(nn.Module):
    def __init__(self,m=1):
        super().__init__()
        self.m=m
        self.activation=torch.sigmoid
        self.loss_fn=nn.BCELoss()
        self.z=torch.tensor(0.,dtype=torch.float32,requires_grad=False)
    def forward(self, y_pred,y_true):
        # 兩者同組時，算square
        # 兩者不同組時，算margin- distance值，若distance大於margin則不用再拉伸兩者distance
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
        # 扣掉對cosine的margin
        cos_phi = cosine - self.m
        # 將onehot沒選中的類別不套用margin，onehot選中的套用margin     
        one_hot=F.one_hot(label, num_classes=self.ways).to(torch.float32)
        metric = (one_hot * cos_phi) + ((1.0 - one_hot) * cosine)
        # 將輸出對比放大
        metric *= self.s
        return self.loss_fn(metric,label)
    
class ArcMarginLoss(AddMarginLoss):
    def __init__(self, s=32.0, m=0.40,ways=10, easy_margin=False,loss_fn=FocalLoss()):
        super().__init__(s,m,ways,loss_fn)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.eps = 1e-6
    def forward(self, cosine, label=None):
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + self.eps)
        # cos(phi)cos(m)-sin(phi)sin(m)變成cos(phi + m)
        # 這個margin加上去使得角度phi需要更小才能使指定類別在softmax(cos(phi))時最大
        cos_phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # cosine如果不夠大就不用不套用phi margin
            cos_phi = torch.where(cosine > 0, cos_phi, cosine)
        else:
            # 更加嚴格，若cosine(phi)大於margin則套用phi margin規則
            #          若cosine(phi)小於margin則套用cosine margin規則
            cos_phi = torch.where(cosine > self.th, cos_phi, cosine - self.mm)
            
        # 將onehot沒選中的類別不套用margin，onehot選中的套用margin    
        one_hot=F.one_hot(label, num_classes=self.ways).to(torch.float32)
        metric = (one_hot * cos_phi) + ((1.0 - one_hot) * cosine)
        # 將輸出對比放大
        metric *= self.s
        return self.loss_fn(metric,label)