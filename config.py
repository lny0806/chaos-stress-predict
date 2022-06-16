class DefaultConfig(object):
    def __init__(self):
        self.model = 'Triangle'

        self.train_data_root = '/home/ubuntu/lny/chaos-new/code/data/StudentLife/learn/new/train-30_1/'
        self.test_data_root = '/home/ubuntu/lny/chaos-new/code/data/StudentLife/learn/new/test-30_1/'

        self.num_workers = 0
        self.hidden = 16
        self.lr = 0.01
        self.lr_decay = 0.95
        self.weight_decay = 1e-4
        self.use_gpu = True

        self.batch_size_karry = 64
        self.max_epoch_karry = 1000

        self.num_classes = 2

        self.input_dim = 6
        self.event_dim = 24
        self.sleep_dim = 6
        self.mood_dim = 24
        self.location_dim = 6
        self.opinion_dim = 8
        self.cognition_dim = 10
        self.history_num = 22
        
        self.memory_d = 16
        self.memory_k = 16

        self.memory_bsd = 16
        self.memory_bsk = 16
        self.personality_dim = 2

