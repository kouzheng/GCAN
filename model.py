import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
import glob
import os
import time
import torch.optim as optim
import argparse
import random
from torch.autograd import Variable
import sys
import pickle as pkl
import networkx as nx
import scipy
from scipy.sparse.linalg import eigsh
from typing import Union, Tuple, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros,ones
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, recall_score

class GCAConv(MessagePassing):
    _alpha: OptTensor
    def __init__(self, in_channels: int, out_channels: int, heads: int,
                 positional_channels:int = 15,
                 concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0., add_self_loops: bool = True,
                 bias: bool = True, share_weights: bool = False, **kwargs):
        super(GCAConv, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.positional_channels = positional_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                            weight_initializer='glorot')
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
        self.att = Parameter(torch.Tensor(1, heads, out_channels))
        self.att2 = Parameter(torch.Tensor(1, heads, self.positional_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self._alpha1 = None
        self._alpha2 = None
        self.bias2 =  Parameter(torch.Tensor(self.positional_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        glorot(self.att)
        glorot(self.att2)
        zeros(self.bias)
        zeros(self.bias2)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                positional: Tensor,
                size: Size = None, return_attention_weights: bool = None):
        # type: (Union[Tensor, PairTensor], Tensor , Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels
        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)  #(N , heads, features)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        assert x_l is not None
        assert x_r is not None
        positional = positional.unsqueeze(dim = 1)
        positional = positional.repeat(1, self.heads, 1)
        x_l = torch.cat((x_l,positional), dim = -1)
        x_r = torch.cat((x_r,positional), dim = -1)
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)
        out_all = self.propagate(edge_index, x=(x_l, x_r), size=size)
        out = out_all[ : , : , :self.out_channels ]
        out2 = out_all[ : , : , self.out_channels:]
        alpha1 = self._alpha1
        self._alpha1 = None
        alpha2 = self._alpha2
        self._alpha2 = None
        if self.concat:
            out = out.reshape(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out += self.bias
        out2 = out2.mean(dim=1)
        out2 += self.bias2

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out , out2

    def message(self, x_j: Tensor, x_i: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha1 = (x[:, :, :self.out_channels] * self.att).sum(dim=-1)
        alpha2 = (x[:, :, self.out_channels:] * self.att2).sum(dim=-1)
        alpha1 = softmax(alpha1, index, ptr, size_i)
        alpha2 = softmax(alpha2, index, ptr, size_i)
        self._alpha1 = alpha1
        self._alpha2 = alpha2
        alpha1= F.dropout(alpha1, p=self.dropout, training=self.training)
        alpha2= F.dropout(alpha2, p=self.dropout, training=self.training)
        return torch.cat((x_j[:, :, :self.out_channels]* alpha2.unsqueeze(-1), x_j[:, :, self.out_channels: ]* alpha1.unsqueeze(-1)) ,dim = -1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GCAN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads , positional_hidden):
        super(GCAN, self).__init__() 
        self.dropout = dropout
        self.alpha = alpha
        self.positional_out = positional_hidden
        self.cj = nn.Conv1d(in_channels=4, out_channels=1 , kernel_size=1)
        self.cj_2 = nn.Conv1d(in_channels=1, out_channels=1 , kernel_size=15 , stride=15)
        self.fc1 = nn.Linear(64, self.positional_out)
        self.fc2 = nn.Linear(self.positional_out, 1)
        self.conv1 = GCAConv(in_channels = 1000,
                                out_channels = nhid,
                                heads = nheads,
                                dropout = self.dropout,
                                negative_slope = self.alpha,
                                positional_channels = self.positional_out
                                )
        self.conv2 = GCAConv(in_channels = nheads * nhid,
                                out_channels = 1,
                                heads = 1,
                                dropout = self.dropout,
                                negative_slope = self.alpha,
                                positional_channels = 1
                                )
    def forward(self, x ,  edge_index, positional):
        x = self.cj(x)
        x = self.cj_2(x)
        x = x.reshape(11374,-1)
        x = F.dropout(x, p = self.dropout , training=self.training)
        positional = F.dropout(positional, p = self.dropout , training=self.training)
        positional = self.fc1(positional)
        x , positional = self.conv1(x, edge_index , positional)
        x = F.relu(x) 
        positional = F.relu(positional)
        x = F.dropout(x, p = self.dropout , training=self.training)
        positional = F.dropout(positional, p = self.dropout , training=self.training)
        positional = self.fc2(positional)
        x ,positional = self.conv2(x, edge_index , positional) 
        positional = positional.reshape(-1)
        x = x.reshape(-1)
        x = torch.sigmoid(positional+x)
        return x

    def forward_conv1(self, x, edge_index, positional):
        x = self.cj(x)
        x = self.cj_2(x)
        x = x.reshape(11374,x.shape[-1])

        positional = F.dropout(positional, p = self.dropout , training=self.training)
        positional = self.fc1(positional)
        x = F.dropout(x, p = self.dropout , training=self.training)
        x , positional = self.conv1(x, edge_index , positional)
        return x


data = np.loadtxt('data/deepwalk_64')
T = data
print(T.shape)
tt=torch.tensor(T)
positional = tt.float()

positional= F.normalize(positional ,dim = 0)
positional = positional.to(torch.float32)
from sklearn.metrics import roc_curve, auc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
positional= positional.to(device)
def train(epoch):
    # trian
    model.train()
    optimizer.zero_grad()
    output = model(features, edge_index , positional)

    weights = torch.tensor([1.0,10.0]).to(device)
    loss_train = F.binary_cross_entropy(output[idx_train], labels[idx_train].float() )
    loss_train.backward()
    optimizer.step()
    # eval
    model.eval()
    output = model(features, edge_index , positional)
    loss_val = F.binary_cross_entropy(output[idx_val], labels[idx_val].float())
    with open("loss_record.txt","a") as f:      
            f.write("\n"+"loss_train:"+str(loss_train.data.item())+
                    "\n"+"loss_val :"+str(loss_val.data.item())) 
    return loss_val.data.item()

from sklearn.metrics import precision_recall_curve, auc
def compute_test():
    model.eval()

    output = model(features,edge_index , positional)
    y_true  = labels[idx_test].cpu()

    fpr, tpr, thresholds = roc_curve(y_true, output[idx_test].cpu().detach())
    roc_auc = auc(fpr, tpr)
    
    # Find the optimal threshold by finding the point with the maximum true positive rate minus false positive rate
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_threshold

    precision, recall, thresholds = precision_recall_curve(y_true,output[idx_test].cpu().detach())
    aupr = auc(recall, precision)

    acc_test = accuracy(output[idx_test], labels[idx_test]  ,  optimal_threshold)
    print(acc_test.item())
    print('optimal_threshold:'+  str(optimal_threshold))
    y_pred = (output > optimal_threshold).type_as(labels)[idx_test].cpu()
    assert len(y_true) == len(y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(roc_auc , precision, recall, f1)
    cm = confusion_matrix(y_true, y_pred)
    cm_1 = cm[0][0]
    cm_2 = cm[0][1]
    cm_3 = cm[1][0]
    cm_4 = cm[1][1]
    false_positives = (y_pred == 1) & (y_true == 0)
    fp_indices = idx_test.cpu()[false_positives].numpy()
    print(cm)
    return acc_test.item() , roc_auc.item(), precision.item(), recall.item(),f1.item(),cm_1.item(),cm_2.item(),cm_3.item(),cm_4.item() , aupr.item()
    
def accuracy(output, labels , roc ):
    preds = (output > roc).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def random_split3(arr, p):                
    l = len(arr)
    l1 = int(l/10)
    arr3 = arr[ l1*p : l1*(p+1)]
    if(p == 9):
        arr3 = arr[ l1*p : ]
    remaining_part = np.concatenate([arr[:l1*p], arr[l1*(p+1):]])
    arr1 = remaining_part[:l1*8]
    arr2 = remaining_part[l1*8: ]
    return torch.tensor(arr1), torch.tensor(arr2), torch.tensor(arr3)

def get_split3(data_y , p):
    label=[]
    label_size = int(max(data_y))+1
    for x in range(label_size):
        label.append([])
    for x in range(len(data_y)):
        label[data_y[x]].append(x)
    train_index=torch.LongTensor([])
    val_index=torch.LongTensor([])
    test_index=torch.LongTensor([])
    for x in range(label_size):
        a , b , c = random_split3(np.array(label[x]) , p)
        train_index = torch.cat((train_index,a))
        val_index = torch.cat((val_index,b))
        test_index = torch.cat((test_index ,c )) 
    return train_index , val_index , test_index


import glob
import os
lr_list = [ 0.01, 0.005 ]
dp_list = [ 0.6 ]
heads_list = [8, 4, 2, 1 ]
hid_list = [ 64  , 32  , 16 , 8]
wd_list = [ 0.001, 0.0005 ]
positional_list = [ 64 ,32 , 16  , 8]

hyper_list = []
    
for nb_heads in heads_list:
    for hidden in hid_list:
        for lr in lr_list:
            for dropout in dp_list:
                for weight_decay in wd_list:
                    for positional_hidden in positional_list:
                        hyper_list.append( [ nb_heads , hidden , lr , dropout , weight_decay , positional_hidden])

for idd in range( 0 , len(hyper_list)):
    all_acc=[]
    for all_times in range(10): 
        if_random_split = True
        hidden = hyper_list[idd][1]
        dropout = hyper_list[idd][3]
        nb_heads = hyper_list[idd][0]
        alpha = 0.2
        lr = hyper_list[idd][2]
        weight_decay = hyper_list[idd][4]
        positional_hidden = hyper_list[idd][5]
        epochs = 1000
        patience = 100
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = epochs + 1
        best_epoch = 0
        edge_index = torch.tensor(np.loadtxt('data/edge').T)
        edge_index = edge_index.to(torch.int)
        data = np.load('data/one_hot_array.npy')
        data = np.transpose(data,(1,0,2))
        features  =  torch.tensor(data)
        labels =  np.loadtxt('data/virus_lab')
        labels = torch.tensor(labels.astype(int))
        features = features.to(torch.float32)
        
        if(if_random_split):
            idx_train , idx_val , idx_test = get_split3(labels , all_times)
        else:
            idx_train , idx_val , idx_test = data.train_mask , data.val_mask ,data.test_mask

        model = GCAN(    nfeat=features.shape[-1], 
                        nhid=hidden, 
                        nclass=int(labels.max()) + 1, 
                        dropout=dropout, 
                        nheads=nb_heads, 
                        alpha=alpha,
                        positional_hidden = positional_hidden
                        )
        optimizer = optim.Adam(model.parameters(), 
                                    lr=lr, 
                                    weight_decay=weight_decay)
        cuda = torch.cuda.is_available()
        if cuda:
                model.to(device)
                features = features.to(device)
                edge_index = edge_index.to(device)
                labels = labels.to(device)
                idx_train = idx_train.to(device)
                idx_val = idx_val.to(device)
                idx_test = idx_test.to(device)
        features,  labels = Variable(features), Variable(labels)         
        
        for epoch in range(epochs):   
            
            loss_values.append(train(epoch))
            torch.save(model.state_dict(), '{}.pkl'.format(epoch))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter == patience:
                break

            files = glob.glob('*.pkl')
            for file in files:
                epoch_nb = int(file.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)
        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)


        model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
        result = compute_test()
        all_acc.append(result)
        formatted_acc = int(np.array(result)[1] * 10000)
        files = glob.glob('*.pkl')
        os.remove(files[0])
    all_acc = np.array(all_acc)
    with open("result.txt","a") as f:      
        f.write("\n"+str(idd)+ ":  "+str(all_acc[:,0].mean()) +"  "+str(all_acc[:,1].mean())+ "  "
                +str(all_acc[:,2].mean())+ "  "+str(all_acc[:,3].mean())+ "  "+str(all_acc[:,4].mean())+"  "+
                str(all_acc[:,5].mean())+"  "+
                str(all_acc[:,6].mean())+"  "+
                str(all_acc[:,7].mean())+"  "+
                str(all_acc[:,8].mean())+"  "+
                str(all_acc[:,9].mean())+"  "+
                str(all_acc[:,0].std())+"  "+
                str(all_acc[:,1].std())+"  "+
                str(all_acc[:,2].std())+"  "+
                str(all_acc[:,3].std())+"  "+
                str(all_acc[:,4].std())+"  "+
                str(all_acc[:,9].std())+"  "+
            "    heads:"+ str(nb_heads) +
            "    hidden: "+ str(hidden)+
            "    dropout: "+str(dropout)+
            "    lr:"+str(lr)+
            "    weight_decay:"+str(weight_decay) )
