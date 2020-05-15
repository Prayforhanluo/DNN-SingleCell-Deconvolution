# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:46:47 2019

@author: LuoHan
"""


import numpy as np
import pandas as pd
import os
import torch
import sys
import torch.utils.data as Data
import scanpy as sc
from tqdm import tqdm       
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split


class DataConstruction:
    """
        Data Proprecessing for .h5md file
    """
    def __init__(self, h5md_file, ds, log_transfrom = True, markers = None):
        """
        """
        self.h5md = h5md_file
        self.ds = ds
        self.log_transform = log_transfrom
        self.markers = markers
    
    def DataLoading(self):
        """
        """
        Data = sc.read_h5ad(self.h5md)
        target = Data.obs
        target = target[target.ds == self.ds]
        target = target.iloc[:,:-2]
        samples = target.index
        
        data = Data.to_df()
        data = data.reindex[samples]
        
        self.samples = data.index
        self.cells = target.columns
        
        #Scale Data
        if self.log_transform == True:
            data = np.log2(data + 1)
        scaler = pp.MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        
        if self.markers is not None:
            data = data[self.markers]


def DataLoading(sc_count_file, bulk_count_file, bulk_prop_file):
    """
    """
    sc_count = pd.read_csv(sc_count_file, index_col = 0)
    
    #dispertion gene select
    Dispertion = np.log2(sc_count.var(axis = 1) / sc_count.mean(axis = 1))
    Dispertion_gene = Dispertion[Dispertion > 0.5].index
    
    Bulk = pd.read_csv(bulk_count_file, index_col = 0)
    Prop = pd.read_csv(bulk_prop_file, index_col = 0)
    Bulk = Bulk.reindex(Dispertion_gene)
    
    return Bulk, Prop


class LightNet(torch.nn.Module):
    """
        First Light Net in Scaden
    """
    def __init__(self, n_classes, featuresize, epoch = 1000, lr = 0.0001):
        """
        """
        super(LightNet, self).__init__()
        self.n_classes = n_classes
        self.featuresize = featuresize
        self.hidden_layer = [256, 128, 64, 32]
        self.batchsize = 128
        self.epoch = epoch
        self.lr = lr
        ##Net
        self.net = torch.nn.Sequential(
                    torch.nn.Linear(self.featuresize,self.hidden_layer[0]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_layer[0], self.hidden_layer[1]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_layer[1],self.hidden_layer[2]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_layer[2],self.hidden_layer[3]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_layer[3],self.n_classes),
                    torch.nn.Softmax(dim=1))

class MiddleNet(torch.nn.Module):
    """
        Second Middle Net in Scaden
    """
    def __init__(self, n_classes, featuresize, epoch=1000, lr = 0.0001):
        super(MiddleNet, self).__init__()
        self.n_classes = n_classes
        self.featuresize = featuresize
        self.hidden_layer = [512,256,128,64]
        self.batchsize = 128
        self.epoch = 1000
        self.lr = lr
        #Net
        self.net = torch.nn.Sequential(
                    torch.nn.Linear(self.featuresize, self.hidden_layer[0]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_layer[0], self.hidden_layer[1]),
                    torch.nn.Dropout(0.3),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_layer[1], self.hidden_layer[2]),
                    torch.nn.Dropout(0.2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_layer[2],self.hidden_layer[3]),
                    torch.nn.Dropout(0.1),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_layer[3],self.n_classes),
                    torch.nn.Softmax(dim=1))

class HeavyNet(torch.nn.Module):
    """
        Third Heavy Net in Scaden
    """
    def __init__(self, n_classes, featuresize, epoch=1000, lr = 0.0001):
        super(HeavyNet, self).__init__()
        self.n_classes = n_classes
        self.featuresize = featuresize
        self.hidden_layer = [1024,516,256,128]
        self.batchsize = 128
        self.epoch = 1000
        self.lr = lr
        #Net
        self.net = torch.nn.Sequential(
                    torch.nn.Linear(self.featuresize, self.hidden_layer[0]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_layer[0], self.hidden_layer[1]),
                    torch.nn.Dropout(0.6),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_layer[1], self.hidden_layer[2]),
                    torch.nn.Dropout(0.3),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_layer[2],self.hidden_layer[3]),
                    torch.nn.Dropout(0.1),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_layer[3],self.n_classes),
                    torch.nn.Softmax(dim=1))

def train(model, train_X, train_y):
    """
    """
    torch.set_num_threads(4)
    DataSet = Data.TensorDataset(train_X, train_y)
    Dataloader = Data.DataLoader(dataset=DataSet,
                                 batch_size=model.batchsize,
                                 shuffle=True,
                                 num_workers=0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = model.lr)
    pbar = tqdm(range(model.epoch))
    loss_fn = torch.nn.MSELoss()
    Loss_value = []
    
    for _ in pbar:
        for step, (batch_x,batch_y) in enumerate(Dataloader):
            batch_y_pred = model.net(batch_x)
            loss = loss_fn(batch_y_pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Loss_value.append(loss)
        if (_+1) % 100 == 0:
            report = 'Step : {}, Loss : {}'.format(_+1, loss)
            pbar.write(report)
        
    return Loss_value

def ComputeCCC(y_pred, y):
    """
    """
    cor = np.corrcoef(y, y_pred)[0][1]
    
    var_true = y.var().item()
    var_pred = y_pred.var().item()
    
    mean_true = y.mean().item()
    mean_pred = y_pred.mean().item()
    
    std_true = y.std().item()
    std_pred = y_pred.std().item()
    
    top = 2 * cor * std_true *std_pred
    bottom = var_true + var_pred + (mean_true - mean_pred)**2

    return top / bottom



if __name__ == '__main__':
    sc_count_file = sys.argv[1]
    Bulk_file = sys.argv[2]
    Prop_file = sys.argv[3]
    
    Bulk, Prop = DataLoading(sc_count_file, Bulk_file, Prop_file)
    Prop = Prop.divide(Prop.sum(axis=1), axis=0)
    
    X, y = Bulk.values.T, Prop.values
    
    #normalize
    X = np.log2(X + 1)
    scaler = pp.MinMaxScaler()
    X = scaler.fit_transform(X.T).T
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, 
                                                        random_state = 42)
    
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    
    LN = LightNet(featuresize=X_train.size()[1], n_classes=y_train.size()[1])
    MN = MiddleNet(featuresize=X_train.size()[1], n_classes=y_train.size()[1])
    HN = HeavyNet(featuresize=X_train.size()[1], n_classes=y_train.size()[1])
    
    #Use cuda to train.
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        X_train_cuda = X_train.cuda()
        y_train_cuda = y_train.cuda()
        X_test_cuda = X_test.cuda()
        y_test_cuda = y_test.cuda()
        LN_cuda = LN.cuda()
        MN_cuda = MN.cuda()
        HN_cuda = HN.cuda()
        
        print('Trainning Light Net ...')
        LN_loss_value = train(LN_cuda, X_train_cuda, y_train_cuda)
        y_pred = LN_cuda.net(X_test_cuda).cpu()
        score = np.array([ComputeCCC(y_pred[i].data, y_test[i]) for i in range(len(y_pred))])
        print('Light Net Score : {}'.format(score.mean()))
        
        print('Trainning Middle Net ...')    
        MN_loss_value = train(MN_cuda, X_train_cuda, y_train_cuda)
        y_pred = MN_cuda.net(X_test_cuda)
        score = np.array([ComputeCCC(y_pred[i].data, y_test[i]) for i in range(len(y_pred))])
        print('Middle Net Score : {}'.format(score.mean()))
        
        HN_loss_value = train(HN_cuda, X_train_cuda, y_train_cuda)
        y_pred = HN_cuda.net(X_test_cuda)
        score = np.array([ComputeCCC(y_pred[i].data, y_test[i]) for i in range(len(y_pred))])
        print('Heavy Net Score : {}'.format(score.mean()))
    #Use cpu to train
    else:
        print('Trainning Light Net ...')
        LN_loss_value = train(LN, X_train, y_train)
        y_pred = LN.net(X_test)
        score = np.array([ComputeCCC(y_pred[i].data, y_test[i]) for i in range(len(y_pred))])
        print('Light Net Score : {}'.format(score.mean()))
        
        print('Trainning Middle Net ...')    
        MN_loss_value = train(MN, X_train, y_train)
        y_pred = MN.net(X_test_cuda)
        score = np.array([ComputeCCC(y_pred[i].data, y_test[i]) for i in range(len(y_pred))])
        print('Middle Net Score : {}'.format(score.mean()))
        
        HN_loss_value = train(HN, X_train, y_train)
        y_pred = HN.net(X_test)
        score = np.array([ComputeCCC(y_pred[i].data, y_test[i]) for i in range(len(y_pred))])
        print('Heavy Net Score : {}'.format(score.mean()))
