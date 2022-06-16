import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import Tensor as Tensor
from torch.nn import Parameter as P

from config import DefaultConfig
from .MLP import MLP

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
class TriangleCell(nn.Module):
    def __init__(self,memory_k,memory_d, batch_first=True):
        super(TriangleCell, self).__init__()

        self.opt = DefaultConfig()


        self.num_layers = 1
        self.batch_first = batch_first
        self.memory_k = memory_k
        self.memory_d = memory_d

        self.memory_bsk = self.opt.memory_bsk
        self.memory_bsd = self.opt.memory_bsd

        self.mlp_out = 8
        

        self.input_weights_w = nn.Linear(self.opt.input_dim+self.opt.memory_d, 4 * self.opt.hidden)
        self.input_weights_s = nn.Linear(self.opt.sleep_dim+self.opt.memory_d, 4 * self.opt.hidden)
        self.input_weights_m = nn.Linear(self.opt.mood_dim+self.opt.memory_d, 4 * self.opt.hidden)
        self.input_weights_l = nn.Linear(self.opt.location_dim+self.opt.memory_d, 4 * self.opt.hidden)
        self.hidden_weights_w = nn.Linear(self.opt.hidden, 4 * self.opt.hidden)
        self.hidden_weights = nn.Linear(self.opt.hidden, 4 * self.opt.hidden)
  

        self.we_w = nn.Linear(self.mlp_out,memory_d)
        self.we_wbs = nn.Linear(self.opt.hidden,self.memory_bsd)
        self.we_bbs = nn.Linear(self.opt.hidden,self.memory_bsd)

        self.wk_w = nn.Linear(self.opt.event_dim,memory_d)
        self.wk_wbs = nn.Linear(self.opt.hidden,self.memory_bsd)
        self.wk_bbs = nn.Linear(self.opt.hidden,self.memory_bsd)

        self.wa_w = nn.Linear(self.mlp_out,memory_d)

        self.wa_wbs = nn.Linear(self.opt.hidden,self.memory_bsd)
        self.wa_bbs = nn.Linear(self.opt.hidden,self.memory_bsd)


        self.wl_wbs = nn.Linear(self.memory_bsd,self.opt.hidden,bias=False)
        self.wl_sbs = nn.Linear(self.memory_bsd,self.opt.hidden,bias=False)
        self.wl_mbs = nn.Linear(self.memory_bsd,self.opt.hidden,bias=False)
        self.wl_lbs = nn.Linear(self.memory_bsd,self.opt.hidden,bias=False)

        self.wp_wbs = nn.Linear(self.opt.hidden,self.opt.hidden,bias=False)
        self.wp_sbs = nn.Linear(self.opt.hidden,self.opt.hidden,bias=False)
        self.wp_mbs = nn.Linear(self.opt.hidden,self.opt.hidden,bias=False)
        self.wp_lbs = nn.Linear(self.opt.hidden,self.opt.hidden,bias=False)

        self.wq_wbs = nn.Linear(self.memory_bsd,self.opt.hidden,bias=False)
        self.wq_sbs = nn.Linear(self.memory_bsd,self.opt.hidden,bias=False)
        self.wq_mbs = nn.Linear(self.memory_bsd,self.opt.hidden,bias=False)
        self.wq_lbs = nn.Linear(self.memory_bsd,self.opt.hidden,bias=False)

        self.p = 0.6

        self.behavioratt = nn.Linear(self.opt.hidden,1)
        self.mlp_ebs = MLP(self.opt.hidden+self.opt.hidden+self.opt.event_dim,self.mlp_out,self.mlp_out)

         

    def forward(self,stress,inputelem,sleep,mood,event,location):
        batch_size = stress.shape[0]
        if inputelem is None:
            hw_old = torch.randn(batch_size, self.opt.hidden).to(device)
            cw_old = torch.randn(batch_size,self.opt.hidden).to(device)
            hs_old = torch.randn(batch_size,self.opt.hidden).to(device)
            cs_old = torch.randn(batch_size,self.opt.hidden).to(device)
            hm_old = torch.randn(batch_size,self.opt.hidden).to(device)
            cm_old = torch.randn(batch_size,self.opt.hidden).to(device)
            m_old = torch.zeros(batch_size,self.memory_k,self.memory_d).to(device)
            bm_old = torch.zeros(batch_size,self.memory_bsk,self.memory_bsd).to(device)
            sm_old = torch.zeros(batch_size,self.memory_bsk,self.memory_bsd).to(device)
            bsm_old =  self.p*bm_old + (1-self.p)*sm_old
            r_w_old = torch.zeros(batch_size,self.memory_d).to(device)
            r_s_old = torch.zeros(batch_size,self.memory_d).to(device)
            r_m_old = torch.zeros(batch_size,self.memory_d).to(device)
            hl_old = torch.randn(batch_size, self.opt.hidden).to(device)
            cl_old = torch.randn(batch_size,self.opt.hidden).to(device)
            r_l_old = torch.zeros(batch_size,self.memory_d).to(device)

        else:
            hw_old = inputelem[0] 
            cw_old = inputelem[1]
            hs_old = inputelem[2] 
            cs_old = inputelem[3]
            hm_old = inputelem[4] 
            cm_old = inputelem[5]
            m_old = inputelem[6]
            bsm_old = inputelem[7]
            bm_old = inputelem[8]
            sm_old = inputelem[9]
            r_w_old = inputelem[10]
            r_s_old = inputelem[11]
            r_m_old = inputelem[12]
            hl_old = inputelem[13]
            cl_old = inputelem[14]
            r_l_old = inputelem[15]



        stress_in = torch.cat((stress,r_w_old),dim=-1)

        wgates = self.input_weights_w(stress_in) + self.hidden_weights_w(hw_old)
        wingate, wforgetgate, wcellgate, woutgate = wgates.chunk(4, 1) #拆分各个门

        wingate = torch.sigmoid(wingate)
        wforgetgate = torch.sigmoid(wforgetgate)
        wcellgate = torch.tanh(wcellgate)
        woutgate = torch.sigmoid(woutgate)

        cw_new = (wforgetgate * cw_old) + (wingate * wcellgate)
        hw_new = woutgate * cw_new


        sleep_in = torch.cat((sleep,r_s_old),dim=-1)
        sgates = self.input_weights_s(sleep_in) + self.hidden_weights(hs_old)
        singate, sforgetgate, scellgate, soutgate = sgates.chunk(4, 1) #拆分各个门

        singate = torch.sigmoid(singate)
        sforgetgate = torch.sigmoid(sforgetgate)
        scellgate = torch.tanh(scellgate)
        soutgate = torch.sigmoid(soutgate)

        cs_new = (sforgetgate * cs_old) + (singate * scellgate)
        hs_new = soutgate * cs_new


        mood_in = torch.cat((mood,r_m_old),dim=-1)
        mgates = self.input_weights_m(mood_in) + self.hidden_weights(hm_old)
        mingate, mforgetgate, mcellgate, moutgate = mgates.chunk(4, 1) #拆分各个门

        mingate = torch.sigmoid(mingate)
        mforgetgate = torch.sigmoid(mforgetgate)
        mcellgate = torch.tanh(mcellgate)
        moutgate = torch.sigmoid(moutgate)

        cm_new = (mforgetgate * cm_old) + (mingate * mcellgate)
        hm_new = moutgate * cm_new

        location_in = torch.cat((location,r_l_old),dim=-1)
        lgates = self.input_weights_l(location_in) + self.hidden_weights(hl_old)
        lingate, lforgetgate, lcellgate, loutgate = lgates.chunk(4, 1) #拆分各个门
        lingate = torch.sigmoid(lingate)
        lforgetgate = torch.sigmoid(lforgetgate)
        lcellgate = torch.tanh(lcellgate)
        loutgate = torch.sigmoid(loutgate)

        cl_new = (lforgetgate * cl_old) + (lingate * lcellgate)
        hl_new = loutgate * cl_new



        behavior = torch.stack([hs_new,hm_new,hl_new],1)
        behavior_att = F.softmax(self.behavioratt(behavior),dim=1)
        behavior_t = torch.reshape(behavior,(-1,self.opt.hidden,3))
        behavior_out = torch.bmm(behavior_t,behavior_att)
        behavior_new = torch.squeeze(behavior_out)
        

        ebs_cat = torch.cat((hw_new,behavior_new,event),dim=-1)
        ebs = self.mlp_ebs(ebs_cat)



        key_w = torch.tanh(self.wk_w(event))
        e_w = torch.sigmoid(self.we_w(ebs))
        a_w = torch.tanh(self.wa_w(ebs))

        m_old_t = m_old.permute(0,2,1)

        key_w_t = torch.reshape(key_w,(-1,1,self.memory_d))

        

        alpha_w = F.softmax(torch.bmm(key_w_t,m_old_t),dim=2)   
        alpha_w_t = alpha_w
        r_w = torch.bmm(alpha_w_t,m_old)
        r_w = torch.reshape(r_w,(-1,self.memory_d))



        key_s = key_w
        key_m = key_w
        key_l = key_w

        key_s_t = torch.reshape(key_s,(-1,1,self.memory_d))
        alpha_s = F.softmax(torch.bmm(key_s_t,m_old_t),dim=2)
        alpha_s_t = alpha_s
        r_s = torch.bmm(alpha_s_t,m_old)
        r_s = torch.reshape(r_s,(-1,self.memory_d))

        key_m_t = torch.reshape(key_m,(-1,1,self.memory_d))
        alpha_m = F.softmax(torch.bmm(key_m_t,m_old_t),dim=2)
        alpha_m_t = alpha_m
        r_m = torch.bmm(alpha_m_t,m_old)
        r_m = torch.reshape(r_m,(-1,self.memory_d))

        key_l_t = torch.reshape(key_l,(-1,1,self.memory_d))
        alpha_l = F.softmax(torch.bmm(key_l_t,m_old_t),dim=2)
        alpha_l_t = alpha_l
        r_l = torch.bmm(alpha_l_t,m_old)
        r_l = torch.reshape(r_l,(-1,self.memory_d))


        alpha_w = alpha_w.permute(0,2,1)

        e_w = torch.reshape(e_w,(-1,1,e_w.shape[1]))
        a_w = torch.reshape(a_w,(-1,1,a_w.shape[1]))
        unit_tensor = torch.ones(batch_size,self.memory_k,self.memory_d).to(device)
        m_new = m_old*(unit_tensor-torch.bmm(alpha_w,e_w))+torch.bmm(alpha_w,a_w)
        

        key_wbs = torch.tanh(self.wk_wbs(hw_new))
        e_wbs = torch.sigmoid(self.we_wbs(hw_new))
        a_wbs = torch.tanh(self.wa_wbs(hw_new))

        bsm_old_t = bsm_old.permute(0,2,1)
        key_wbs_t = torch.reshape(key_wbs,(-1,1,self.memory_bsd))
        alpha_wbs = F.softmax(torch.bmm(key_wbs_t,bsm_old_t),dim=2)
        alpha_wbs_t = alpha_wbs
        r_wbs = torch.bmm(alpha_wbs_t,bsm_old)
        r_wbs = torch.reshape(r_wbs,(-1,self.memory_bsd))
        g_wbs = torch.sigmoid(self.wp_wbs(cw_new)+self.wq_wbs(r_wbs))
        hw_new = woutgate * (cw_new+g_wbs*self.wl_wbs(r_wbs))

        alpha_wbs = alpha_wbs.permute(0,2,1)
        e_wbs = torch.reshape(e_wbs,(-1,1,e_wbs.shape[1]))
        a_wbs = torch.reshape(a_wbs,(-1,1,a_wbs.shape[1]))

        key_bbs = torch.tanh(self.wk_bbs(behavior_new))
        e_bbs = torch.sigmoid(self.we_bbs(behavior_new))
        a_bbs = torch.tanh(self.wa_bbs(behavior_new))

        key_bbs_t = torch.reshape(key_bbs,(-1,1,self.memory_bsd))
        alpha_bbs = F.softmax(torch.bmm(key_bbs_t,bsm_old_t),dim=2)
        
        key_sbs = key_bbs
        key_mbs = key_bbs
        key_lbs = key_bbs

        key_sbs_t = torch.reshape(key_sbs,(-1,1,self.memory_bsd))
        alpha_sbs = F.softmax(torch.bmm(key_sbs_t,bsm_old_t),dim=2)
        alpha_sbs_t = alpha_sbs
        r_sbs = torch.bmm(alpha_sbs_t,bsm_old)
        r_sbs = torch.reshape(r_sbs,(-1,self.memory_bsd))
        g_sbs = torch.sigmoid(self.wp_sbs(cs_new)+self.wq_sbs(r_sbs))
        hs_new = soutgate * (cs_new + g_sbs*self.wl_sbs(r_sbs))

        key_mbs_t = torch.reshape(key_mbs,(-1,1,self.memory_bsd))
        alpha_mbs = F.softmax(torch.bmm(key_mbs_t,bsm_old_t),dim=2)
        alpha_mbs_t = alpha_mbs
        r_mbs = torch.bmm(alpha_mbs_t,bsm_old)
        r_mbs = torch.reshape(r_mbs,(-1,self.memory_bsd))
        g_mbs = torch.sigmoid(self.wp_mbs(cm_new)+self.wq_mbs(r_mbs))
        hm_new = moutgate * (cm_new + g_mbs*self.wl_mbs(r_mbs)) 

        key_lbs_t = torch.reshape(key_lbs,(-1,1,self.memory_bsd))
        alpha_lbs = F.softmax(torch.bmm(key_lbs_t,bsm_old_t),dim=2)
        alpha_lbs_t = alpha_lbs
        r_lbs = torch.bmm(alpha_lbs_t,bsm_old)
        r_lbs = torch.reshape(r_lbs,(-1,self.memory_bsd))
        g_lbs = torch.sigmoid(self.wp_lbs(cl_new)+self.wq_lbs(r_lbs))
        hl_new = loutgate * (cl_new + g_lbs*self.wl_lbs(r_lbs)) 

        alpha_bbs = alpha_bbs.permute(0,2,1)
        e_bbs = torch.reshape(e_bbs,(-1,1,e_bbs.shape[1]))
        a_bbs = torch.reshape(a_bbs,(-1,1,a_bbs.shape[1]))


        unit_tensor_bs = torch.ones(batch_size,self.memory_bsk,self.memory_bsd).to(device)
        bm_new = bm_old*(unit_tensor_bs-torch.bmm(alpha_wbs,e_wbs))+torch.bmm(alpha_wbs,a_wbs)
        sm_new = sm_old*(unit_tensor_bs-torch.bmm(alpha_bbs,e_bbs))+torch.bmm(alpha_bbs,a_bbs)
        bsm_new =  self.p*bm_new + (1-self.p)*sm_new

 
        return hw_new,cw_new,hs_new,cs_new,hm_new,cm_new,m_new,bsm_new,bm_new,sm_new,r_w,r_s,r_m,hl_new,cl_new,r_l



