import torch
import os 
import os.path as osp 
import numpy as np
from scipy.io import loadmat
import gc
from typing import Tuple

class Dataset():
    """This class reads the dataset from the files."""
    
    
    
    def __init__(self,filename1="BPSK.mat",filename2="QPSK.mat",
    filename3="8PSK.mat",filename4="16QAM.mat",filename5="64QAM.mat",filename6="256QAM.mat",data_dir1:str="C:/Users/Lenovo/OFDM-OTFS-modulation-recognition/dataset/Gauss"):
        #storage path for datasets

        self.filename1=filename1
        self.filename2=filename2
        self.filename3=filename3
        self.filename4=filename4
        self.filename5=filename5
        self.filename6=filename6
        
        self.path1=osp.join(data_dir1,self.filename1)
        self.path2=osp.join(data_dir1,self.filename2)
        self.path3=osp.join(data_dir1,self.filename3)
        self.path4=osp.join(data_dir1,self.filename4)
        self.path5=osp.join(data_dir1,self.filename5)
        self.path6=osp.join(data_dir1,self.filename6)
        


        self.X, self.label_mod,self.label_snr = self.load_data()
        self.modulations={
        "['BPSK']":0,
        "['QPSK']":1,
        "['8PSK']":2,
        "['16QAM']":3,
        "['64QAM']":4,
        "['256QAM']":5,
       #generate labels

    }
    
        gc.collect()

    def load_data(self):
        #BPSK_
        data_bpsk=loadmat(self.path1)
        data_bpsk=data_bpsk['dataset']
        
        

        data_qpsk=loadmat(self.path2)
        data_qpsk=data_qpsk['dataset']
         
        data_8psk=loadmat(self.path3)
        data_8psk=data_8psk['dataset']

        data_16qam=loadmat(self.path4)
        data_16qam=data_16qam['dataset']

        data_64qam=loadmat(self.path5)
        data_64qam=data_64qam['dataset']

        data_256qam=loadmat(self.path6)
        data_256qam=data_256qam['dataset']






        #num_data=4096*16
        num_data=2000*16 #number of samples in each class

        X,label_mod,label_snr=[],[],[]
        for i in range(num_data):
            data=data_bpsk[0][i][0][0]
        #[1], [4] and [5] are fixed to 0
        #[2] represents the number of samples taken, the range is 0 - the number of samples.
        #[3] represents the removed label or data, range 0-1 (0 for data, 1 for label).
        #[6] represents the category of the removed tag (0 is the modulation method, 1 is the signal-to-noise ratio)
            data=torch.FloatTensor(data)
            data=data.unsqueeze(0)
            X.append(data)
            lab_mod=data_bpsk[0][i][1][0][0][0]
            lab_mod=str(lab_mod)
            label_mod.append(lab_mod)

            lab_snr=data_bpsk[0][i][1][0][0][1]
            lab_snr=int(lab_snr)
            label_snr.append(lab_snr)
        # print(label_snr[-1])
        # print(label_snr[0])
        # print(label_snr)
        # print(lab_mod)
        # eixt()
        for i in range(num_data):
            data=data_qpsk[0][i][0][0]
            data=torch.FloatTensor(data)
            data=data.unsqueeze(0)
            X.append(data)
            lab_mod=data_qpsk[0][i][1][0][0][0]
            lab_mod=str(lab_mod)
            label_mod.append(lab_mod)

            lab_snr=data_qpsk[0][i][1][0][0][1]
            lab_snr=int(lab_snr)
            label_snr.append(lab_snr)

        

        for i in range(num_data):
            data=data_8psk[0][i][0][0]
            data=torch.FloatTensor(data)
            data=data.unsqueeze(0)
            X.append(data)
            lab_mod=data_8psk[0][i][1][0][0][0]
            lab_mod=str(lab_mod)
            label_mod.append(lab_mod)

            lab_snr=data_8psk[0][i][1][0][0][1]
            lab_snr=int(lab_snr)
            label_snr.append(lab_snr)

        

        for i in range(num_data):
            data=data_16qam[0][i][0][0]
            data=torch.FloatTensor(data)
            data=data.unsqueeze(0)
            X.append(data)
            lab_mod=data_16qam[0][i][1][0][0][0]
            lab_mod=str(lab_mod)
            label_mod.append(lab_mod)

            lab_snr=data_16qam[0][i][1][0][0][1]
            lab_snr=int(lab_snr)
            label_snr.append(lab_snr)
        


        for i in range(num_data):
            data=data_64qam[0][i][0][0]
            data=torch.FloatTensor(data)
            data=data.unsqueeze(0)
            X.append(data)
            lab_mod=data_64qam[0][i][1][0][0][0]
            lab_mod=str(lab_mod)
            label_mod.append(lab_mod)

            lab_snr=data_64qam[0][i][1][0][0][1]
            lab_snr=int(lab_snr)
            label_snr.append(lab_snr)
        
        

        for i in range(num_data):
            data=data_256qam[0][i][0][0]
            data=torch.FloatTensor(data)
            data=data.unsqueeze(0)
            X.append(data)
            lab_mod=data_256qam[0][i][1][0][0][0]
            lab_mod=str(lab_mod)
            label_mod.append(lab_mod)

            lab_snr=data_256qam[0][i][1][0][0][1]
            lab_snr=int(lab_snr)
            label_snr.append(lab_snr)
        # print(X[-1])
        # print(label_mod[-1])
        # print(label_snr[-1])
        # print(X)
        # print(label_mod)
        # print(label_snr)

        
        
        #print(X[-1])
        #print(label[-1])
        

        
        
        X = np.vstack(X)
    
        return X,label_mod,label_snr
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x=self.X[idx]
        mod=self.label_mod[idx]
        snr=self.label_snr[idx]
        label = self.modulations[mod]
        x= torch.FloatTensor(x)
        label=torch.tensor(label, dtype=torch.long)
        #x=x.unsqueeze(0)
        return x,label,snr #the dimension of X is 1×2×1024
    def __len__(self) -> int:
        return self.X.shape[0]

#data=Dataset()
dataset = Dataset()

# 方式1：直接打印 load_data 返回的结果（但注意 load_data 已经在 __init__ 中被调用了）
# 如果你还想重新加载数据，可以显式调用：
# X, label_mod, label_snr = dataset.load_data()
# print("X shape:", X.shape)
# print("First 5 labels:", label_mod[:5])
# print("First 5 SNRs:", label_snr[:50])



        


        
        










        
        





        