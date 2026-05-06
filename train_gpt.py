import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader,random_split
import numpy as np
from model import GPT
from utils import Averager,count_acc
from tqdm import tqdm
from Underwater_dataset import Dataset
import os




class Train():
    """
    Perform training, validation and testing.
    Output recognition accuracy at different signal-to-noise ratios during validation phase.
    """
    def __init__(self):
       
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset=Dataset()
        self.split_train=0.6
        self.split_val=0.3
        self.train_set,self.val_set,self.test_set=self.load_dataset()
        self.batch_size=32 #一次给的数据大小
        self.train_loader=dataloader.DataLoader(dataset=self.train_set,batch_size=self.batch_size,shuffle=True)
        self.val_loader=dataloader.DataLoader(dataset=self.val_set,batch_size=self.batch_size,shuffle=True)
        self.test_loader=dataloader.DataLoader(dataset=self.test_set,batch_size=self.batch_size,shuffle=True)
        self.model=GPT().to(self.device)
        self.optim=torch.optim.Adam(self.model.parameters(),lr=0.001)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optim, step_size=25, gamma=0.1)
        self.model_path = "./model/gpt_underwater.pth"

        self.num_epochs=1
        self.snrs=[-10,-8,-6,-4,-2,0,2,4,6,8,10]
        #self.snrs=[0,2,4,6,8,10,14,16,18]


    def load_dataset(self):
        #torch.cuda.manual_seed(99)
        total=len(self.dataset)
        length=[int(total*self.split_train)]
        length.append(int(total*self.split_val))
        length.append(total-length[0]-length[1])
        print("Splitting into {} train and {} val and {} test".format(length[0], length[1],length[2]))
        train_set,val_set,test_set=random_split(self.dataset,length)
        return train_set,val_set,test_set

    
    def snr_load(self,snr):
        self.snr_data=[]
        self.snr_label=[]
        for i in range(len(self.test_set)):
            a,b,c=self.test_set[i]
            if c==snr:
                self.snr_data.append(a)
                self.snr_label.append(b)
        if len(self.snr_data) == 0:
            return None
        self.snr_data=torch.stack(self.snr_data)
        self.snr_label=torch.tensor(self.snr_label, dtype=torch.long)
        test_dataset=torch.utils.data.TensorDataset(self.snr_data,self.snr_label)


        return test_dataset

    def evaluate_loader(self, loader):
        loss_averager = Averager()
        acc_averager = Averager()

        self.model.eval()
        with torch.no_grad():
            for data, label, *_ in loader:
                data = data.to(self.device)
                label = label.to(self.device)
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                loss_averager.add(loss.item())
                acc_averager.add(acc)

        return loss_averager.item(), acc_averager.item()

    def train(self):
        os.makedirs("./model", exist_ok=True)
        for epoch in range(1,self.num_epochs+1):
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            self.model.train()

            for i,(data,label,_) in enumerate(tqdm(self.train_loader)):
                data=data.to(self.device)
                label=label.to(self.device)               
                
                self.optim.zero_grad()
                logits=self.model(data)
                loss=F.cross_entropy(logits,label)
                acc=count_acc(logits,label)
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)
                loss.backward()
                self.optim.step()
            self.lr_scheduler.step()
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()
            print('Epoch {}, train, Loss={:.4f} Acc={:.4f}'.format(epoch, train_loss_averager, train_acc_averager))

            val_loss_averager, val_acc_averager = self.evaluate_loader(self.val_loader)
            print('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, val_loss_averager, val_acc_averager))
        torch.save(self.model.state_dict(), self.model_path)

    
        pretrained_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(pretrained_dict)
        self.model.eval()
        
        for snr in self.snrs:
            test_dataset=self.snr_load(snr)
            if test_dataset is None:
                print(f"snr={snr}, no test samples")
                continue
            snr_test_loader=dataloader.DataLoader(dataset=test_dataset,batch_size=self.batch_size,shuffle=True)
            test_loss_averager, test_acc_averager = self.evaluate_loader(snr_test_loader)
            print(f"snr={snr}, Acc={test_acc_averager}")


                    
    def test(self):
        model=GPT()
        
        pretrained_dict = torch.load(self.model_path, map_location=self.device)
       
        model.load_state_dict(pretrained_dict)
        model=model.to(self.device)
        model.eval()
        


        test_loss_averager = Averager()
        test_acc_averager = Averager()
        with torch.no_grad():
            for i,(data,label,_) in enumerate(tqdm(self.test_loader)):
                data=data.to(self.device)
                label=label.to(self.device)
                logits=model(data)
                loss=F.cross_entropy(logits,label)
                acc=count_acc(logits,label)
                test_loss_averager.add(loss.item())
                test_acc_averager.add(acc)
            
        test_loss_averager = test_loss_averager.item()
        test_acc_averager = test_acc_averager.item()
        print('Test, Loss={:.4f} Acc={:.4f}'.format(test_loss_averager, test_acc_averager))






if __name__ == '__main__':
    Train().train()
    # Train().test()




