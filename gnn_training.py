import math
import time
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import scipy.sparse as sp
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from utils.load_noisydata import load_noisydata
import argparse
import torch.nn.functional as F
# from models.gnn_models import GAT,GCN
from models.model_unnamed4 import to_edge_index,accuracy,calculate_classification_metrics
from torch_geometric.nn import MessagePassing, GCNConv

parser = argparse.ArgumentParser(
    description="PyTorch implementation of model")
parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='dataset used.Cora,CiteSeer,Photo,PubMed,ogbn-arxiv')
parser.add_argument('--noise_type', type=str, default='uniform', help='uniform,pair_noise,class_confusion,feature_based')
parser.add_argument('--epochs', type=int, default=5000,
                    help='number of epochs to train ')
parser.add_argument('--graph_noise', type=float, default=0.0,
                    help='rate of the graph_noise')
parser.add_argument('--label_noise', type=float, default=0.0,
                    help='rate of the label_noise')
parser.add_argument('--feature_noise', type=float, default=0.0, 
                    help='rate of the feature_noise')
parser.add_argument('--hidden_size', type=float, default=128,
                    help='dimension of hidden representation')
parser.add_argument('--patience', type=int, default=100, help='early stop.')

parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, self_loop=True, device=None):
        # monitor_resource(operation="GCN - 初始化前", prefix="PRE", mode="start")
        super(GCN, self).__init__()

        self.device = device
        self.gc1 = GCNConv(nfeat, nhid, bias=True, add_self_loops=self_loop)  # 第一层GCN
        self.gc2 = GCNConv(nhid, nclass, bias=True, add_self_loops=self_loop)  # 第二层GCN
        self.dropout = dropout
        # monitor_resource(operation="GCN - 初始化完成（创建2层GCNConv）", prefix="POST", mode="end")

    def forward(self, x, edge_index, edge_weight):
        # monitor_resource(operation="GCN.forward - 前向传播前", prefix="PRE", mode="start")
        # 第一层GCN
        # monitor_resource(operation="GCN.forward - 第一层GCN完成", prefix="MID1", mode="start")
        x1 = F.relu(self.gc1(x, edge_index, edge_weight))  # 第一层GCN+激活（高显存）
        # monitor_resource(operation="GCN.forward - 第一层GCN完成", prefix="MID1", mode="end")
        x1 = F.dropout(x1, self.dropout, training=self.training)  # Dropout
        # 第二层GCN
        # monitor_resource(operation="GCN.forward - 第二层GCN完成", prefix="MID2", mode="start")
        x1 = self.gc2(x1, edge_index, edge_weight)  # 第二层GCN（高显存）
        # monitor_resource(operation="GCN.forward - 第二层GCN完成", prefix="MID2", mode="end")
        # PyTorch内置函数
        # x1=x1*100
        # x1 = F.softmax(x1, dim=1)
        # monitor_resource(operation="GCN.forward - 前向传播完成", prefix="POST", mode="start")
        x1 = F.normalize(x1, p=2, dim=-1)  # 归一化
        # monitor_resource(operation="GCN.forward - 前向传播完成", prefix="POST", mode="end")
        return x1

    def initialize(self):
        # monitor_resource(operation="GCN.initialize - 参数重置前", prefix="PRE", mode="start")
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        # monitor_resource(operation="GCN.initialize - 参数重置完成", prefix="POST", mode="end")


class gnn_fit():
    def __init__(self,args):
        super(gnn_fit, self).__init__()
        self.args=args
        self.best_acc_pred_val = 0
    
    def train(self,epoch):
        t1 = time.time()
        args.model.train()
        args.optimizer.zero_grad()                                                            # 梯度清零
        output = args.model(args.features, args.edge_index, args.edge_weight)                     # features:(2708, 1433)   adj:(2708, 2708)
        loss_train = F.cross_entropy(output[args.train_index], args.targets_oneHot1[args.train_index])  # 损失函数
        acc_train = accuracy(output[args.train_index], args.targets_c_oneHot[args.train_index])                       # 计算准确率
        loss_train.backward()                                                            # 反向传播
        args.optimizer.step()                                                                 # 更新梯度

        
        args.model.eval()
        output = args.model(args.features, args.edge_index, args.edge_weight)                     # features:(2708, 1433)   adj:(2708, 2708)

        loss_val = F.cross_entropy(output[args.val_index], args.targets_oneHot1[args.val_index])
        acc_val = accuracy(output[args.val_index], args.targets_c_oneHot[args.val_index])
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t1))
        if acc_val >= self.best_acc_pred_val:
            print(1)
            self.best_acc_pred_val = acc_val
            self.predictor_model_weigths1 = deepcopy(self.args.model.state_dict())
            better_inex = 1
        t=time.time()-t1
        return acc_val.item(),t
    
        
    def test(self):
        args.model.eval()
        intime = []
        # for i in range(2000):
        #     t1= time.time()
        output = args.model(args.features, args.edge_index, args.edge_weight)                     # features:(2708, 1433)   adj:(2708, 2708)
        #     t2 = time.time()
        #     itime= t2 - t1
        #     intime.append(itime)
        #     print(i)
        # print("final_time:{:.4f}".format(np.mean(np.array(intime))),"{:.4f}".format(np.std(np.array(intime))))
        loss_test = F.cross_entropy(output[args.test_index], args.targets_oneHot1[args.test_index])
        acc_test = accuracy(output[args.test_index], args.targets_c_oneHot[args.test_index])
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))
        pre,rec,f1=calculate_classification_metrics(output[args.test_index], args.targets_c_oneHot[args.test_index])
        return acc_test.item(),pre.item(),rec.item(),f1.item()
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = 16                                     # 定义隐藏层数
    dropout = 0.5
    lr = 0.01 
    weight_decay = 5e-4
    fastmode = 'store_true'
    acc_list=[]
    pre_list=[]
    rec_list=[]
    f1_list=[]
    for seed in range(2020, 2024):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        (args.train_index, args.val_index, args.test_index, args.train_label, args.val_label, args.test_label,
                args.features, args.adj, args.true_train_label, args.num_classes, args.labels) = load_noisydata(
                    noise_type=args.noise_type,
                    dataset=args.dataset,
                    graph_noise=args.graph_noise,
                    label_noise=args.label_noise,
                    feature_noise=args.feature_noise
                )
        
        args.train_label = args.train_label.values.reshape(-1)
        args.train_label = torch.from_numpy(args.train_label)
        args.train_label = torch.tensor(args.train_label, dtype=torch.long, device='cuda:0')

        args.train_index = torch.tensor(args.train_index, dtype=torch.long, device='cuda:0')
        args.train_index_clean = args.train_index.clone()
        args.val_index = torch.tensor(args.val_index, dtype=torch.long, device='cuda:0')
        args.test_index = torch.tensor(args.test_index, dtype=torch.long, device='cuda:0')
        args.ft_size = args.features.shape[1]
        args.features = torch.tensor(args.features, dtype=torch.float32, device='cuda:0')
        args.labels = torch.tensor(args.labels, dtype=torch.long, device='cuda:0')
        args.input_size = args.features[args.train_index].shape[1]
        args.features1=args.features.clone()
        args.features2 = args.features.clone()
        args.adj = args.adj.tocoo()
        args.num_edge = args.adj.nnz
        args.labels_c = args.labels.clone()
        args.idx_unlabel = torch.LongTensor(list(set(range(args.features.shape[0])) - set(args.train_index))).to('cuda:0')
        args.labels_noise = args.labels.clone()
        args.labels_noise[args.train_index] = args.train_label
        ones = torch.sparse.torch.eye(args.num_classes).to(args.device)
        args.targets_oneHot1 = ones.index_select(torch.tensor(0, device='cuda:0'), torch.tensor(args.labels_noise, device='cuda:0'))
        args.targets_oneHot2=args.targets_oneHot1.clone()
        args.pre_true_label = args.targets_oneHot1.clone()
        args.targets_c_oneHot = ones.index_select(torch.tensor(0, device='cuda:0'),
                                                torch.tensor(args.labels_c, device='cuda:0'))

        args.labels_oneHot = args.targets_oneHot1.clone()

        # n=0
        # for i in range(2708):
        #     n+=np.count_nonzero(args.adj.toarray()[i])
        # print(n)
        # args.adj = torch.tensor(args.adj.toarray(), device='cuda:0')
        args.edge_index, args.edge_weight = to_edge_index(args.adj)
        args.model=GCN(args.features.shape[1],args.hidden_size,args.targets_oneHot1.shape[1],dropout=0.5).to(args.device)

        args.optimizer = torch.optim.Adam(args.model.parameters(),lr=lr, weight_decay=weight_decay)
        
        gnn_train=gnn_fit(args)
        # if device:                                          # 数据放在cuda上
        #     args.model.cuda()
        #     args.features = args.features.cuda()
        #     args.adj = args.adj.cuda()
        #     args.labels = args.labels.cuda()
        #     idx_train = idx_train.cuda()
        #     idx_val = idx_val.cuda()
        #     idx_test = idx_test.cuda()
        epochs = 5000
        time_list=[]
        for epoch in range(epochs):
            val_acc,t = gnn_train.train(epoch)
            time_list.append(t)
            if val_acc >= gnn_train.best_acc_pred_val:
                # print(1)
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!')
                break
        print("training_time:{:.4f}".format(np.mean(np.array(time_list))),"{:.4f}".format(np.std(np.array(time_list))))
        print("Optimization Finished!")

        print("picking the best model according to validation performance")

        args.model.load_state_dict(gnn_train.predictor_model_weigths1)
        acc, pre, rec, f1 = gnn_train.test()
        acc_list.append(acc*100)
        pre_list.append(pre*100)
        rec_list.append(rec*100)
        f1_list.append(f1)
    print("final_acc:{:.2f}".format(np.mean(np.array(acc_list))),"{:.2f}".format(np.std(np.array(acc_list))))
    print("final_pre:{:.2f}".format(np.mean(np.array(pre_list))),"{:.2f}".format(np.std(np.array(pre_list))))
    print("final_rec:{:.2f}".format(np.mean(np.array(rec_list))),"{:.2f}".format(np.std(np.array(rec_list))))
    print("final_f1:{:.2f}".format(np.mean(np.array(f1_list))),"{:.2f}".format(np.std(np.array(f1_list))))
    

if __name__ == "__main__":
    main()
