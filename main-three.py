import models
import torch
from data import StressDataThree
from config import DefaultConfig
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score,classification_report,confusion_matrix
import torch.nn.utils.rnn as rnn_utils
import math
from torch.optim import lr_scheduler


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") 





def test(model,dataloader,lenn,batch_size,writer,epoch,weight_CE):
    model.eval()
    sum_acc = 0.
    sum_loss = 0.
    sum_y_, sum_target = np.asarray([]),np.asarray([])
    criterion = torch.nn.CrossEntropyLoss(weight_CE)

    for i, (x,index,label,event,cognition,opinion,location,mood,sleep,personality,future_event) in tqdm(enumerate(dataloader), total = lenn/batch_size):
        x = x.type(torch.FloatTensor)
        x = x.to(device)
        label = label.to(device)
        event = event.to(device)
        cognition = cognition.to(device)
        opinion = opinion.to(device)
        location = location.to(device)
        mood = mood.to(device)
        sleep = sleep.to(device)
        personality = personality.to(device)
        future_event = future_event.to(device)
        score = model((x,event,cognition,opinion,location,mood,sleep,personality,future_event))

        y_ = torch.argmax(score,dim = 1)
        loss = criterion(score,label)
        correct = sum(y_.cpu().numpy()==label.cpu().numpy())/batch_size
        sum_acc += correct
        sum_loss += loss.data
        
        sum_y_ =  np.concatenate((sum_y_,y_.cpu().numpy()),axis=0)
        sum_target = np.concatenate((sum_target,label.cpu().numpy()),axis=0)
        
    f1 = metrics(sum_target,sum_y_)
    le = lenn/batch_size
    writer.add_scalar("/log/test_acc",sum_acc/le,epoch)
    writer.add_scalar("/log/test_f1",f1,epoch)
    writer.add_scalar("/log/test_loss",sum_loss/le,epoch)
    print("test correct {} ".format(sum_acc/le))
    print("test f1 {} ".format(f1))
    print('test loss {}'.format(sum_loss/le))
    model.train()

def metrics(y_true,y_pred):

    print('accuracy',accuracy_score(y_true, y_pred))
    print('precision',precision_score(y_true, y_pred, average='macro'))
    print('recall',recall_score(y_true, y_pred, average='macro'))
    f1 = f1_score(y_true, y_pred, average='macro')
    print('macro ',f1)

    print (confusion_matrix(y_true, y_pred))

    target_names = ['1','2','3']
    print(classification_report(y_true, y_pred, target_names=target_names))
    return f1

def train_karry():
    opt = DefaultConfig()
    model = getattr(models, opt.model)()
    model.to(device)
    train_data = StressDataThree(opt.train_data_root,opt)
    test_data = StressDataThree(opt.test_data_root,opt)
    train_dataloader = DataLoader(
        train_data,
        opt.batch_size_karry,
        shuffle = True,
        num_workers = opt.num_workers,
        )
    test_dataloader = DataLoader(
        test_data,
        opt.batch_size_karry,
        shuffle = False,
        num_workers = opt.num_workers,
        )
    weight_CE = torch.FloatTensor([2,1,1])
    weight_CE = weight_CE.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_CE)
    lr = opt.lr
    
    optimizer = torch.optim.Adam(model.parameters(),
        lr = lr,
        weight_decay = opt.weight_decay)
    writer = SummaryWriter()
    cnt=0

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150,200,300,400,500,700,900], gamma=0.9)
    sum_acc, sum_loss = 0.,0.
    for epoch in range(opt.max_epoch_karry):
        scheduler.step()
        for i, (x,index,label,event,cognition,opinion,location,mood,sleep,personality,future_event) in tqdm(enumerate(train_dataloader),total=len(train_data)/opt.batch_size_karry):
            x = x.type(torch.FloatTensor)
            x = x.to(device)
            label = label.to(device)
            event = event.to(device)
            cognition = cognition.to(device)
            opinion = opinion.to(device)
            location = location.to(device)
            mood = mood.to(device)
            sleep = sleep.to(device)
            personality = personality.to(device)
            future_event = future_event.to(device)
            
            optimizer.zero_grad()

            score = model((x,event,cognition,opinion,location,mood,sleep,personality,future_event))
            loss = criterion(score,label)

            loss.backward()
            optimizer.step()
            y_ = torch.argmax(score,dim = 1)
            correct = sum(y_.cpu().numpy()==label.cpu().numpy())/opt.batch_size_karry
            sum_acc += correct
            sum_loss += loss.data
            cnt +=1

        lenn = math.ceil(len(train_data)/opt.batch_size_karry)
        writer.add_scalar("/log/train_acc",sum_acc/lenn,epoch)
        writer.add_scalar("/log/train_loss",sum_loss/lenn,epoch)
        print("train correct {} ".format(sum_acc/lenn))
        print("train loss {} ".format(sum_loss/lenn))
        sum_acc, sum_loss = 0., 0.
        test(model,test_dataloader,len(test_data),opt.batch_size_karry,writer,epoch,weight_CE)

    writer.close()



def main():
    train_karry()
    

if __name__ == '__main__':
    main()


    