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
from models.gnn_models import GAT,GCN,GCN_LP
from models.model_unnamed4 import to_edge_index,accuracy
from sklearn.metrics import roc_auc_score
from models.model_unnamed5 import extract_subgraph_from_edge_index, negative_sampling,get_link_labels

parser = argparse.ArgumentParser(
    description="PyTorch implementation of model")
parser.add_argument('--dataset', type=str, default='CiteSeer', help='dataset used.Cora,CiteSeer,Photo,PubMed')
parser.add_argument('--noise_type', type=str, default='uniform', help='uniform,pair_noise,class_confusion,feature_based')
parser.add_argument('--epochs', type=int, default=5000,
                    help='number of epochs to train ')
parser.add_argument('--graph_noise', type=float, default=0.1,
                    help='rate of the graph_noise')
parser.add_argument('--label_noise', type=float, default=0.1,
                    help='rate of the label_noise')
parser.add_argument('--feature_noise', type=float, default=0.3, 
                    help='rate of the feature_noise')
parser.add_argument('--hidden_size', type=float, default=128,
                    help='dimension of hidden representation')
parser.add_argument('--patience', type=int, default=100, help='early stop.')

parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

class gnn_fit():
    def __init__(self,args):
        super(gnn_fit, self).__init__()
        self.args=args
        self.best_acc_pred_val = 0
    
    def train(self,epoch):
        t = time.time()
        args.model.train()
        args.optimizer.zero_grad()    
        
        
                                                                
        output = args.model.encode(args.features, args.edge_pos_train)
        link_logits = args.model.decode(output, args.edge_pos_train, args.edge_neg_train)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(args.edge_pos_train, args.edge_neg_train).to("cuda:0")
        loss_train = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        # loss_train = F.cross_entropy(output[args.train_index], args.targets_oneHot1[args.train_index])  # 损失函数
        acc_train=roc_auc_score(link_labels.cpu().detach().numpy(), link_probs.cpu().detach().numpy())                       # 计算准确率
        loss_train.backward()                                                            # 反向传播
        args.optimizer.step()                                                                 # 更新梯度

        
        args.model.eval()
        output = args.model.encode(args.features, args.edge_pos_train)
        link_logits = args.model.decode(output, args.edge_pos_val, args.edge_neg_val)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(args.edge_pos_val, args.edge_neg_val).to("cuda:0")
        loss_val = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        # loss_train = F.cross_entropy(output[args.train_index], args.targets_oneHot1[args.train_index])  # 损失函数
        acc_val=roc_auc_score(link_labels.cpu().detach().numpy(), link_probs.cpu().detach().numpy()) 

        output = args.model.encode(args.features, args.edge_pos_train)
        link_logits = args.model.decode(output, args.edge_pos_test, args.edge_neg_test)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(args.edge_pos_test, args.edge_neg_test).to("cuda:0")
        loss_test = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        # loss_train = F.cross_entropy(output[args.train_index], args.targets_oneHot1[args.train_index])  # 损失函数
        acc_test=roc_auc_score(link_labels.cpu().detach().numpy(), link_probs.cpu().detach().numpy()) 


        print('Epoch: {:04d}'.format(epoch+1),
            # 'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            # 'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            # 'loss_test: {:.4f}'.format(loss_test.item()),
            'acc_test: {:.4f}'.format(acc_test.item()),
            'time: {:.4f}s'.format(time.time() - t))
        if acc_val >= self.best_acc_pred_val:
            print(1)
            self.best_acc_pred_val = acc_val
            self.predictor_model_weigths1 = deepcopy(self.args.model.state_dict())
            better_inex = 1
        return acc_val.item()
    
        
    def test(self):
        args.model.eval()
        
        output = args.model.encode(args.features, args.edge_pos_train)
        link_logits = args.model.decode(output, args.edge_pos_test, args.edge_neg_test)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(args.edge_pos_test, args.edge_neg_test).to("cuda:0")
        loss_test = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        # loss_train = F.cross_entropy(output[args.train_index], args.targets_oneHot1[args.train_index])  # 损失函数
        acc_test=roc_auc_score(link_labels.cpu().detach().numpy(), link_probs.cpu().detach().numpy()) 
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = 16                                     # 定义隐藏层数
    dropout = 0.5
    lr = 0.001 
    weight_decay = 5e-4
    fastmode = 'store_true'
    acc_list=[]
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
        args.adj = torch.tensor(args.adj.toarray(), device='cuda:0')
        args.edge_index, args.edge_weight = to_edge_index(args.adj)
        args.edge_pos_train=extract_subgraph_from_edge_index(edge_index=args.edge_index, node_indices=args.train_index)
        args.edge_pos_val=extract_subgraph_from_edge_index(edge_index=args.edge_index, node_indices=args.val_index)
        args.edge_pos_test=extract_subgraph_from_edge_index(edge_index=args.edge_index, node_indices=args.test_index)
        args.edge_neg_train = negative_sampling(edge_index=args.edge_pos_train,num_nodes=args.train_index.shape[0],num_neg_samples=args.edge_pos_train.shape[1])
        args.edge_neg_val = negative_sampling(edge_index=args.edge_pos_val,num_nodes=args.val_index.shape[0],num_neg_samples=args.edge_pos_val.shape[1])
        args.edge_neg_test = negative_sampling(edge_index=args.edge_pos_test,num_nodes=args.test_index.shape[0],num_neg_samples=args.edge_pos_test.shape[1])
        args.model=GCN_LP(args.features.shape[1],args.hidden_size,64).to(args.device)

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
        for epoch in range(epochs):
            val_acc = gnn_train.train(epoch)

            if val_acc >= gnn_train.best_acc_pred_val:
                # print(1)
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!')
                break
        print("Optimization Finished!")

        print("picking the best model according to validation performance")

        args.model.load_state_dict(gnn_train.predictor_model_weigths1)
        
        acc_list.append(gnn_train.test()*100)
    print("final_acc:{:.4f}".format(np.mean(np.array(acc_list))),"std:{:.4f}".format(np.std(np.array(acc_list))))

if __name__ == "__main__":
    main()
