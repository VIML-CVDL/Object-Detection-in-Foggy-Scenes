import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random
from torch.nn import functional as F

def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu

def distance(a, b):
    return torch.sqrt(((a - b) ** 2).sum()).unsqueeze(0)

def distance_batch(a, b):
    bs, _ = a.shape
    result = distance(a[0], b)
    for i in range(bs-1):
        result = torch.cat((result, distance(a[i], b)), 0)

    return result

def multiply(x): #to flatten matrix into a vector
    return functools.reduce(lambda x,y: x*y, x, 1)

def flatten(x):
    """ Flatten matrix into a vector """
    count = multiply(x.size())
    return x.resize_(count)

def index(batch_size, x):
    idx = torch.arange(0, batch_size).long()
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)

def MemoryLoss(memory):

    m, d = memory.size()
    memory_t = torch.t(memory)
    similarity = (torch.matmul(memory, memory_t))/2 + 1/2 # 30X30
    identity_mask = torch.eye(m).cuda()
    sim = torch.abs(similarity - identity_mask)

    return torch.sum(sim)/(m*(m-1))

def prepare_mask(gt_boxes, indices, query):
    masked_query_list = []
    bs, h, w, d = query.size()
    for index in indices:
        query_tmp = query
        resize_gt = gt_boxes[:,index[1],:]
        resize_gt[0][0] = resize_gt[0][0]/1200*75
        resize_gt[0][1] = resize_gt[0][1]/600*37
        resize_gt[0][2] = resize_gt[0][2]/1200*75
        resize_gt[0][3] = resize_gt[0][3]/600*37
        resize_gt[0][0] = torch.floor(resize_gt[0][0])
        resize_gt[0][2] = torch.ceil(resize_gt[0][2])
        resize_gt[0][1] = torch.floor(resize_gt[0][1])
        resize_gt[0][3] = torch.ceil(resize_gt[0][3])
        resize_gt = torch.ceil(resize_gt)
        h1 = (int(resize_gt[0][1].item())-1) if (int(resize_gt[0][1].item())-1)>0 else 0
        h2 = (int(resize_gt[0][3].item())-1) if (int(resize_gt[0][3].item())-1)>0 else 0
        w1 = (int(resize_gt[0][0].item())-1) if (int(resize_gt[0][0].item())-1)>0 else 0
        w2 = (int(resize_gt[0][2].item())-1) if (int(resize_gt[0][2].item())-1)>0 else 0
        if h1==h2:
            if h2+1<37:
                h2=h2+1
            else:
                h1=h1-1
        if w1==w2:
            if w2+1<75:
                w2=w2+1
            else:
                w1=w1-1
        cropped_image = query_tmp[:, h1:h2, w1:w2, :]
        masked_query_list.append(cropped_image)
    return masked_query_list


class _theta(nn.Module):
    def __init__(self,dim):
        super(_theta,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=3, padding=1, stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,256,kernel_size=3, padding=1,stride=1,bias=False)
        self.Conv3=nn.Conv2d(256,128,kernel_size=3, padding=1,stride=1,bias=False)
        self.Conv4=nn.Conv2d(128,64,kernel_size=3, padding=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)

    def forward(self,x):
        x=self.reLu(self.Conv1(x))
        x=self.reLu(self.Conv2(x))
        x=self.reLu(self.Conv3(x))
        x=self.Conv4(x)
        return x

class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim):
        super(Memory, self).__init__()
        # Constants
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.theta = _theta(feature_dim)
        self.thetak = _theta(feature_dim)

    def hard_neg_mem(self, mem, i):
        similarity = torch.matmul(mem,torch.t(self.keys_var))
        similarity[:,i] = -1
        _, max_idx = torch.topk(similarity, 1, dim=1)


        return self.keys_var[max_idx]

    def random_pick_memory(self, mem, max_indices):

        m, d = mem.size()
        output = []
        for i in range(m):
            flattened_indices = (max_indices==i).nonzero()
            a, _ = flattened_indices.size()
            if a != 0:
                number = np.random.choice(a, 1)
                output.append(flattened_indices[number, 0])
            else:
                output.append(-1)

        return torch.tensor(output)

    def get_update_query(self, mem, max_indices, update_indices, score, query, train):
        m, d = mem.size()
        query_update = torch.zeros((m,d)).cuda()
        max_indices = max_indices.cpu()
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1)==i)
            a, _ = idx.size()
            #ex = update_indices[0][i]
            if a != 0:
                #idx = idx[idx != ex]
                query_update[i] = torch.sum(((score[idx,i] / torch.max(score[:,i])) *query[idx].squeeze(1)), dim=0)
#                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
            else:
                query_update[i] = 0

        return query_update

    def get_score(self, mem, query):
        bs, h,w,d = query.size()
        m, d = mem.size()
        query = query.contiguous().view(bs*h*w, d)# (b X h X w) X m
        query = query.unsqueeze(0)
        query = query.repeat(m,1,1)
        mem = mem.unsqueeze(1)
        mem = mem.repeat(1,bs*h*w,1)
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        cosscore = cos(query,mem)

        score_query = F.softmax(cosscore, dim=0)
        score_memory = F.softmax(cosscore, dim=1)

        return score_query,score_memory

    def forward(self, query, keys, train=True, gt_boxes=None, indices=None):

        batch_size, dims,h,w = query.size() # b X d X h X w
        query = F.normalize(query, dim=1)
        query = query.permute(0,2,3,1) # b X h X w X d

        #train
        if train:
            #gathering loss
            gathering_loss = self.gather_loss(query, keys, train)
            #spreading_loss
            spreading_loss = self.spread_loss(query, keys, train)
            # read
            updated_query, softmax_score_query,softmax_score_memory = self.read(query, keys)

            if len(indices) != 0:
                masked_query_list = prepare_mask(gt_boxes, indices, query)
                updated_memory = keys
                for masked_query in masked_query_list:
                    #update
                    updated_memory = self.update(masked_query, updated_memory, train)
            else:
                updated_memory = keys

            #attenmap
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)

            query = query.permute(0,3,1,2)
            updated_query = updated_query.permute(0,3,1,2)
            attenmask = cos(self.thetak(updated_query),self.theta(query))
            cfeature = torch.mul(attenmask, query)
            return updated_memory, cfeature, gathering_loss, spreading_loss

        #test
        else:
            # read
            updated_query, softmax_score_query,softmax_score_memory = self.read(query, keys)
            #update
            updated_memory = keys
            #attenmap
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            query = query.permute(0,3,1,2)
            updated_query = updated_query.permute(0,3,1,2)
            attenmask = cos(self.thetak(updated_query),self.theta(query))
            cfeature = torch.mul(attenmask, query)
            return updated_memory, cfeature

    def update(self, query, keys,train):

        batch_size, h,w,dims = query.size()
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        query_reshape  = query.contiguous().view(batch_size*h*w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)

        memory_hat = torch.matmul(softmax_score_memory.detach(), query_reshape)
        #query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query, query_reshape,train)
        updated_memory = F.normalize(memory_hat + keys, dim=1)
        # top-1 update
        #query_update = query_reshape[updating_indices][0]
        #updated_memory = F.normalize(query_update + keys, dim=1)

        return updated_memory.detach()


    def pointwise_gather_loss(self, query_reshape, keys, gathering_indices, train):
        n,dims = query_reshape.size() # (b X h X w) X d
        loss_mse = torch.nn.MSELoss(reduction='none')

        pointwise_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return pointwise_loss

    def spread_loss(self,query, keys, train):
        batch_size, h,w,dims = query.size() # b X h X w X d

        loss = torch.nn.TripletMarginLoss(margin=1.0)

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size*h*w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=0)
        gathering_indices = gathering_indices.contiguous().view(batch_size*h*w, 2)

        #1st, 2nd closest memories
        pos = keys[gathering_indices[:,0]]
        neg = keys[gathering_indices[:,1]]

        spreading_loss = loss(query_reshape,pos.detach(), neg.detach())

        return spreading_loss

    def gather_loss(self, query, keys, train):

        batch_size, h,w,dims = query.size() # b X h X w X d

        loss_mse = torch.nn.MSELoss()

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size*h*w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=0)
        gathering_indices = gathering_indices.contiguous().view(batch_size*h*w, 1)
        gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return gathering_loss

    def read(self, query, updated_memory):
        batch_size, h,w,dims = query.size() # b X h X w X d

        # 20*37*75*1
        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query)
        softmax_score_query = softmax_score_query.permute(1,0)
        feat_hat = torch.matmul(softmax_score_query.detach(), updated_memory)
        feat_hat = query.contiguous().view(batch_size, h, w, dims)
        return feat_hat, softmax_score_query, softmax_score_memory
