import numpy as np
import torch

from torch.utils import data as tud

class FewShotSampler(tud.Sampler):
    def __init__(self,ds_object,classes,shots,repeats=16,shuffle=True):
        # 儲存參數進object
        self.ds_object=ds_object
        self.classes=classes
        self.shots=shots
        self.repeats=repeats # 在dataset總數少的時候，為了可以組batch，會把多個epoch的資料疊在一起
        self.class_samples=self.get_samples()
        self.shuffle=shuffle
        
    def get_samples(self):
        # 將dataset每個內容拿出來，若class與指定clas相同則存起來
        new_target_list=np.random.permutation([*enumerate(self.ds_object.targets)])

        indices=[]
        for c in self.classes:
            count=self.shots
            for ii,yy in new_target_list:
                # 若已抽到 shot個則停止
                if count==0:
                    break
                # 還未抽到則檢查該label是否為指定class
                if yy==self.ds_object.class_to_idx[c]:
                    # 若是，則加入列表
                    indices.append(ii)
                    count-=1
        return np.repeat(indices,self.repeats)
    
    def __len__(self):
        return len(self.class_samples)
    def __iter__(self):
        if self.shuffle:
            return iter(np.random.permutation(self.class_samples))
        return iter(self.class_samples)
    
class MetaLearningDataset(tud.IterableDataset):
    def __init__(self,ds_object,classes,ways,shots,repeats=20):
        super().__init__()
        self.ds_object=ds_object
        self.classes=classes
        self.maps={v:i for i,v in enumerate(self.classes)} # 原本class順序與新順序對照表
        self.n_classes=len(classes)
        
        self.ways=ways
        self.shots=shots
        self.repeats=repeats
        
        self.class_sample_list=self.get_class_samples() #此處每個class分開做，之後要依照class抽取較方便
        self.class_sample_len=[*map(len,self.class_sample_list)]
        
    def get_class_samples(self):
        # 將class分開儲存
        indices_c=[[] for _ in range(self.n_classes)]
        
        # 屬於每個class的index儲存在對應class的list中
        for ii,yy in enumerate(self.ds_object.targets):
            yy=self.ds_object.classes[yy]
            if yy in self.classes:
                indices_c[self.maps[yy]].append(ii)
        return indices_c
    
    def get_x_in_ds(self,idx):
        return self.ds_object[idx][0]
    def get_y_in_ds(self,idx):
        return self.ds_object[idx][1]
    
    def __iter__(self):
        # 跟few shot sampler那邊一樣，需要好多資料才能組成batch，所以可重複抽數次
        for _ in range(self.repeats):
            #先決定好task抽取順序
            order=np.random.permutation(self.n_classes)
            #再決定好每個task中query對應的label，數字在0~WAYS-1 之間
            y_queries=[np.random.choice(range(self.ways), size=1, replace=False)[0] 
                  for _ in range(self.n_classes//self.ways)]
            
            # 接著輪迴抽task，共抽TOTAL_CLASSES/WAYS次，去掉小數點
            for task_id,y_query in enumerate(y_queries):
                # 每次有數個被抽到的class，取出對應的sample list
                picked=[self.class_sample_list[tt] for tt in order[self.ways*task_id:self.ways*(task_id+1)]]
                
                tmp=[] #用一個空list裝x 的sample的x
                # 對於每個piked
                for ii,id_in_class in enumerate(picked):
                    if ii==y_query:
                        # 若該class的序號剛好是是y_query，則除了SHOT個support data外還要多抽一個query data
                        sample_id=np.random.choice(id_in_class,size=self.shots+1, replace=False)
                    else:
                        # 如果該序號不是y_query，則抽SHOT個
                        sample_id=np.random.choice(id_in_class,size=self.shots, replace=False)
                    # 每個class抽到的東西先用stack 疊起來，會多一個axis在前面，每個class 維度為[SHOT(+1),CH,W,H]
                    tmp.append(torch.stack([*map(self.get_x_in_ds,sample_id)],dim=0))
                    # 有特殊model可能會利用到y，但大多數metric learning用不到
                    
                # Support那邊就把所有的class資料除了query那個以外全部cascade起來，不會多一個axis，維度為[WAY*SHOT,CH,W,H]
                x_support=torch.cat([tmp[ii][:self.shots] for ii in range(self.ways)],dim=0)
                
                # Query沿著y_query去抓他最後一個抽取出來的data
                x_query = tmp[y_query][-1:]
                
                # 最後concatenate起來，x維度為[WAY*SHOT+1,CH,W,H]，y只有一個一開始抽好的值
                yield torch.cat([x_support, x_query], dim=0),torch.tensor(y_query)