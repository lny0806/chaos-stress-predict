import os
from torch.utils import data
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences 
import re
from torchtext.data import Field, BucketIterator
import torch.nn.utils.rnn as rnn_utils
import torch



class StressDataTwo(data.Dataset):
    def __init__(self, root, opt, transforms=None, train=True, test=False):
        super(StressDataTwo, self).__init__()
        self.root = root
        self.samples = [os.path.join(root,sample) for sample in os.listdir(root)]
        self.opt = opt


    def __getitem__(self,index):
        sample_path = self.samples[index]
        orbit = np.loadtxt(sample_path+'/stress.txt')
        subjectid = sample_path.split('/')[-1]
        event = np.loadtxt(sample_path+'/event.txt')[:-1]
        ff = np.loadtxt(sample_path+'/event.txt')[-1]
        future_event = np.asarray(ff,dtype=np.float32)
        cognition = np.loadtxt(sample_path+'/cognition-latest.txt', dtype='float32')
        opinion = np.loadtxt(sample_path+'/opinion-latest.txt', dtype='float32')
        location = np.loadtxt(sample_path+'/location.txt')[:-1]
        mood = np.loadtxt(sample_path+'/mood.txt')[:-1]
        sleep = np.loadtxt(sample_path+'/sleep.txt')[:-1]


        personality = np.loadtxt(sample_path+'/personality.txt', dtype='float32')
        a = [0,3]
        personality_new = personality[a]

        orbit = np.asarray(orbit)
        label = round(orbit[-1][-1])
        if label == -2 or label == -1:
            label = 0
        elif label == 1 or label == 2 or label == 3:
            label = 1
        

        orbit  = pad_sequences(orbit, maxlen=self.opt.input_dim, dtype='float32', padding='post', truncating='pre', value=0.)
        src = np.round(orbit[:-1])
        src_t = src.transpose()
        src_t = pad_sequences(src_t, maxlen=self.opt.history_num, dtype='float32', padding='pre', truncating='pre', value=0.)
        x = src_t.transpose()

        event_t = event.transpose()
        event_t = pad_sequences(event_t, maxlen=self.opt.history_num, dtype='float32', padding='pre', truncating='pre', value=0.)
        event_final = event_t.transpose()


        location  = pad_sequences(location, maxlen=self.opt.input_dim, dtype='float32', padding='post', truncating='pre', value=0.)
        location_t = location.transpose()
        location_t = pad_sequences(location_t, maxlen=self.opt.history_num, dtype='float32', padding='pre', truncating='pre', value=0.)
        location_final = location_t.transpose()

        mood_t = mood.transpose()
        mood_t = pad_sequences(mood_t, maxlen=self.opt.history_num, dtype='float32', padding='pre', truncating='pre', value=0.)
        mood_final = mood_t.transpose()

        sleep  = pad_sequences(sleep, maxlen=self.opt.input_dim, dtype='float32', padding='post', truncating='pre', value=0.)        
        sleep_t = sleep.transpose()
        sleep_t = pad_sequences(sleep_t, maxlen=self.opt.history_num, dtype='float32', padding='pre', truncating='pre', value=0.)
        sleep_final = sleep_t.transpose()


        return x,index,torch.tensor(label),event_final,cognition,opinion,location_final,mood_final,sleep_final,personality_new,future_event


    def __len__(self):
        return len(self.samples)
