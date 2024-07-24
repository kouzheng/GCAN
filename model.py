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
import numpy as np
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
from typing import Union, Tuple, Optional
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, recall_score

class GCAConv(MessagePassing):
    _alpha: OptTensor
    def __init__(self, in_channels: int, out_channels: int, heads: int,
                 topu_channels:int = 15,
                 concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0., add_self_loops: bool = True,
                 bias: bool = True, share_weights: bool = False, **kwargs):
        super(GCAConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.topu_channels = topu_channels
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

        self.att2 = Parameter(torch.Tensor(1, heads, self.topu_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha1 = None
        self._alpha2 = None

        self.bias2 =  Parameter(torch.Tensor(self.topu_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        glorot(self.att)
        glorot(self.att2)
        zeros(self.bias)
        zeros(self.bias2)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                topu: Tensor,
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
        # print("topu", topu.shape)
        topu = topu.unsqueeze(dim = 1)
        # print("topu_sq", topu.shape)
        topu = topu.repeat(1, self.heads, 1)
        # print("topu", topu.shape)
        # print("x_l", x_l.shape)
        x_l = torch.cat((x_l,topu), dim = -1)
        x_r = torch.cat((x_r,topu), dim = -1)

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

        # propagate_type: (x: PairTensor)
        out_all = self.propagate(edge_index, x=(x_l, x_r), size=size)
        # print("out_all",out_all.shape)

        out = out_all[ : , : , :self.out_channels ]

        out2 = out_all[ : , : , self.out_channels:]

        alpha1 = self._alpha1
        self._alpha1 = None

        alpha2 = self._alpha2
        self._alpha2 = None

        if self.concat:
            out = out.reshape(-1, self.heads * self.out_channels)
            # print("outv2",out.shape)
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

        # print("result",torch.cat((x_j[:, :, :self.out_channels]* alpha2.unsqueeze(-1), x_j[:, :, self.out_channels: ]* alpha1.unsqueeze(-1)) ,dim = -1).shape )
        return torch.cat((x_j[:, :, :self.out_channels]* alpha2.unsqueeze(-1), x_j[:, :, self.out_channels: ]* alpha1.unsqueeze(-1)) ,dim = -1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GCAN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads , topu_hidden):
        super(GCAN, self).__init__() 
        self.dropout = dropout
        self.alpha = alpha
        self.topu_out = topu_hidden
        self.cj = nn.Conv1d(in_channels=4, out_channels=1 , kernel_size=1)
        self.cj_2 = nn.Conv1d(in_channels=1, out_channels=1 , kernel_size=15 , stride=15)
        self.fc1 = nn.Linear(64, self.topu_out)
        self.fc2 = nn.Linear(self.topu_out, 1)
        self.conv1 = GCAConv(in_channels = 1000,
                                out_channels = nhid,
                                heads = nheads,
                                dropout = self.dropout,
                                negative_slope = self.alpha,
                                topu_channels = self.topu_out
                                )
        self.conv2 = GCAConv(in_channels = nheads * nhid,
                                out_channels = 1,
                                heads = 1,
                                dropout = self.dropout,
                                negative_slope = self.alpha,
                                topu_channels = 1
                                )
        
    def forward(self, x ,  edge_index, topu):
        x = self.cj(x)
        x = self.cj_2(x)
        x = x.reshape(11374,-1)
        x = F.dropout(x, p = self.dropout , training=self.training)
        topu = F.dropout(topu, p = self.dropout , training=self.training)

        topu = self.fc1(topu)
        x , topu = self.conv1(x, edge_index , topu)

        x = F.relu(x) 
        topu = F.relu(topu)
        x = F.dropout(x, p = self.dropout , training=self.training)
        topu = F.dropout(topu, p = self.dropout , training=self.training)

        topu = self.fc2(topu)
        x ,topu = self.conv2(x, edge_index , topu) 
        topu = topu.reshape(-1)
        x = x.reshape(-1)
        x = torch.sigmoid(topu+x)
        return x

    def forward_conv1(self, x, edge_index, topu):
        x = self.cj(x)
        x = self.cj_2(x)
        x = x.reshape(11374,x.shape[-1])

        topu = F.dropout(topu, p = self.dropout , training=self.training)
        topu = self.fc1(topu)
        x = F.dropout(x, p = self.dropout , training=self.training)
        x , topu = self.conv1(x, edge_index , topu)
        return x


data = np.loadtxt('../../../ASH/processed_data/deepwalk_64')
T = data
print(T.shape)
tt=torch.tensor(T)
topu = tt.float()

topu= F.normalize(topu ,dim = 0)
topu = topu.to(torch.float32)
from sklearn.metrics import roc_curve, auc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
topu= topu.to(device)
def train(epoch):
    # trian
    model.train()
    optimizer.zero_grad()
    output = model(features, edge_index , topu)

    weights = torch.tensor([1.0,10.0]).to(device)
    loss_train = F.binary_cross_entropy(output[idx_train], labels[idx_train].float() )
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train] , weight=weights)

    loss_train.backward()
    optimizer.step()
    # eval
    model.eval()
    output = model(features, edge_index , topu)
    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    loss_val = F.binary_cross_entropy(output[idx_val], labels[idx_val].float())
    with open("loss_record.txt","a") as f:      
            f.write("\n"+"loss_train:"+str(loss_train.data.item())+
                    "\n"+"loss_val :"+str(loss_val.data.item()))  
    # return loss_val.data.item()
    return loss_val.data.item()

from sklearn.metrics import precision_recall_curve, auc
def compute_test():
    model.eval()

    output = model(features,edge_index , topu)
    y_true  = labels[idx_test].cpu()

    fpr, tpr, thresholds = roc_curve(y_true, output[idx_test].cpu().detach())
    roc_auc = auc(fpr, tpr)
    
    # Find the optimal threshold by finding the point with the maximum true positive rate minus false positive rate
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    optimal_threshold

    precision, recall, thresholds = precision_recall_curve(y_true,output[idx_test].cpu().detach())
    file_path = 'sample_out.txt'

    # 打开文件，'a'模式表示追加内容到文件末尾
    with open(file_path, 'a') as file:
    # 将数组转换为字符串，使用空格分隔元素
        array_str = ' '.join(map(str, output[idx_test].cpu().detach().numpy()))
        file.write(array_str + '\n')
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
    # with open("fp_indices_2000_2.txt", "a") as file:
    #     fp_indices_str = ', '.join(map(str, fp_indices))
    #     file.write(f"{fp_indices_str}\n")  # Assuming you have a variable run_number
    
    
    # false_positives = (y_pred == 0) & (y_true == 1)
    # fp_indices = idx_test.cpu()[false_positives].numpy()
    # with open("fn_indices_2000.txt", "a") as file:
    #     fp_indices_str = ', '.join(map(str, fp_indices))
    #     file.write(f"{fp_indices_str}\n")  # Assuming you have a variable run_number
    
    
    # false_positives = (y_pred == 1) & (y_true == 1)
    # fp_indices = idx_test.cpu()[false_positives].numpy()
    # with open("tp_indices_2000.txt", "a") as file:
    #     fp_indices_str = ', '.join(map(str, fp_indices))
    #     file.write(f"{fp_indices_str}\n")  # Assuming you have a variable run_number

    
    # false_positives = (y_pred == 0) & (y_true == 0)
    # fp_indices = idx_test.cpu()[false_positives].numpy()
    # with open("tn_indices_2000.txt", "a") as file:
    #     fp_indices_str = ', '.join(map(str, fp_indices))
    #     file.write(f"{fp_indices_str}\n")  # Assuming you have a variable run_number
    
    print(cm)
    return acc_test.item() , roc_auc.item(), precision.item(), recall.item(),f1.item(),cm_1.item(),cm_2.item(),cm_3.item(),cm_4.item() , aupr.item()
    
def accuracy(output, labels , roc ):
    preds = (output > roc).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

import random
import random
def random_split(arr):
    # random.seed(123)
    random.shuffle(arr)  # 随机打乱原始数组
    arr1 = arr[:600]      # 取前20个元素作为第一个数组
    arr2 = arr[600:800]    # 取20到50个元素作为第二个数组
    arr3 = arr[800:]      # 剩下的元素作为第三个数组
    return torch.tensor(arr1), torch.tensor(arr2), torch.tensor(arr3)


def random_split2(arr):
    # random.seed(111)
    random.shuffle(arr)  # 随机打乱原始数组                  
    l = len(arr)
    l1 = int(l/10)
    arr1 = arr[ : l1*8]      # 取前20个元素作为第一个数组
    arr2 = arr[l1*8 : l1*9]    # 取20到50个元素作为第二个数组
    arr3 = arr[l1*9 : ]      # 剩下的元素作为第三个数组
    return torch.tensor(arr1), torch.tensor(arr2), torch.tensor(arr3)


def get_split(data_y):
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
        a , b , c = random_split(np.array(label[x]))
        train_index = torch.cat((train_index,a))
        val_index = torch.cat((val_index,b))
        test_index = torch.cat((test_index ,c )) 
    return train_index , val_index , test_index

def get_split2(data_y):
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
        a , b , c = random_split2(np.array(label[x]))
        train_index = torch.cat((train_index,a))
        val_index = torch.cat((val_index,b))
        test_index = torch.cat((test_index ,c )) 
    return train_index , val_index , test_index

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
lr_list = [ 0.01 ]
dp_list = [ 0.6 ]
heads_list = [ 4, 2, 1 ]
hid_list = [ 64  , 32  , 16]
wd_list = [ 0.001 ]
topu_list = [ 64 ,32 , 16  , 8]

hyper_list = []
    
for nb_heads in heads_list:
    for hidden in hid_list:
        for lr in lr_list:
            for dropout in dp_list:
                for weight_decay in wd_list:
                    for topu_hidden in topu_list:
                        hyper_list.append( [ nb_heads , hidden , lr , dropout , weight_decay , topu_hidden])

for idd in range( 0 , len(hyper_list)):
    all_acc=[]
    for all_times in range(10):
        with open("loss_record.txt","a") as f:      
            f.write("\n"+"str(all_times)")  
        if_random_split = True
        hidden = hyper_list[idd][1]
        dropout = hyper_list[idd][3]
        nb_heads = hyper_list[idd][0]
        alpha = 0.2
        lr = hyper_list[idd][2]
        weight_decay = hyper_list[idd][4]
        topu_hidden = hyper_list[idd][5]
        epochs = 1000
        patience = 100
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = epochs + 1
        best_epoch = 0
        edge_index = torch.tensor(np.loadtxt('../../../ASH/processed_data/edge').T)
        edge_index = edge_index.to(torch.int)
        data = np.load('../../../ASH/processed_data/one_hot_array.npy')
        data = np.transpose(data,(1,0,2))
        # data = data [:,:,:100]
        features  =  torch.tensor(data)
        labels =  np.loadtxt('../../../ASH/processed_data/all_lab')
        labels = torch.tensor(labels.astype(int))
        features = features.to(torch.float32)
        
        if(if_random_split):
            idx_train , idx_val , idx_test = get_split3(labels , all_times)
        else:
            idx_train , idx_val , idx_test = data.train_mask , data.val_mask ,data.test_mask

        model = GAT(    nfeat=features.shape[-1], 
                        nhid=hidden, 
                        nclass=int(labels.max()) + 1, 
                        dropout=dropout, 
                        nheads=nb_heads, 
                        alpha=alpha,
                        topu_hidden = topu_hidden
                        )
        # 优化器  
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
            
            # 训练模型并保存loss
            loss_values.append(train(epoch))
            # 保存模型
            torch.save(model.state_dict(), '{}.pkl'.format(epoch))

            # 记录loss最小的epoch
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            # 如果连续patience个epoch，最小Loss都没有变则终止模型训练
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


        # 加载最优模型
        model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
        # print(epoch ,best_epoch ,test_acc_values[best_epoch])
        result = compute_test()
        all_acc.append(result)
        formatted_acc = int(np.array(result)[1] * 10000)
        torch.save(model.state_dict(), 'teacher/head{}_hid{}_t{}_{}.pkl'.format(nb_heads,hidden,topu_hidden,all_times))
        files = glob.glob('*.pkl')
        os.remove(files[0])
    all_acc = np.array(all_acc)
    with open('every_result.txt', 'a') as file:
    # 将二维数组转换为字符串形式
        array_str = '\n'.join(' '.join(str(cell) for cell in row) for row in all_acc)
        # 写入文件
        file.write(array_str)
        file.write('\n')
        file.write('\n')  # 在数组末尾添加一个换行符，以便与后续内容分隔
    with open("kd_mlp_result.txt","a") as f:      
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
