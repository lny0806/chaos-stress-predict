from .BasicModule import BasicModule
from torch import nn
from torch.nn import functional as F
import torch
from config import DefaultConfig
import torchvision.models as models
from .TriangleCell import TriangleCell
from .MLP import MLP
import time
from collections import OrderedDict


class Triangle(BasicModule):

    def __init__(self):
        super(Triangle,self).__init__()
        self.model_name = 'Triangle'
        self.opt = DefaultConfig()


        self.memcell = TriangleCell(memory_k=self.opt.memory_k,memory_d=self.opt.memory_d,batch_first=True)
        self.mlp1 = MLP(self.opt.hidden+self.opt.personality_dim+self.opt.cognition_dim+self.opt.opinion_dim+self.opt.memory_d,8,self.opt.num_classes)
        self.allatt = nn.Sequential(
            nn.Linear(self.opt.hidden,1, bias=False),
        )

        self.key_new = nn.Sequential(
            nn.Linear(self.opt.event_dim,self.opt.memory_d,bias=False),
        )

        
 

    def forward(self, inp):
        x = inp[0]
        event = inp[1]
        cognition = inp[2]
        opinion = inp[3]
        location = inp[4]
        mood = inp[5]
        sleep = inp[6]
        personality = inp[7]
        future_event = inp[8]

        prev_state = None
        for i in range(0,self.opt.history_num):
            xi = x[:,i,:] 
            eventi = event[:,i,:]
            sleepi = sleep[:,i,:]
            moodi = mood[:,i,:]
            locationi = location[:,i,:]   
            prev_state = self.memcell(xi,prev_state,sleepi,moodi,eventi,locationi)
        hw = prev_state[0] 
        hs = prev_state[2]        
        hm = prev_state[4] 
        memory = prev_state[6]
        hl = prev_state[13]
       
        key = torch.tanh(self.key_new(future_event))
        m_t = memory.permute(0,2,1)
        key_t = torch.reshape(key,(-1,1,self.opt.memory_d))
        alpha_w = F.softmax(torch.bmm(key_t,m_t),dim=2)

        r_w_new = torch.bmm(alpha_w,memory)
        r_w_new = torch.reshape(r_w_new,(-1,self.opt.memory_d))


        allout = torch.stack([hw,hs,hm,hl],1)
        all_att = F.softmax(self.allatt(allout),dim=1)
        all_t = torch.reshape(allout,(-1,self.opt.hidden,4))
        finalall = torch.bmm(all_t,all_att)
        finalall_out = torch.squeeze(finalall)


        relation = torch.cat((finalall_out,personality,cognition,opinion,r_w_new),dim=-1)
        relation_out = self.mlp1(relation)

        y = F.softmax(relation_out,dim=1)
        return y
 