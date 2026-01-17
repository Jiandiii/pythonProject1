#%%
import time
import argparse
import numpy as np
import torch
from embedder import embedder
from utils import set_everything, index_to_mask
from copy import deepcopy
from collections import defaultdict
import torch_geometric.utils as tu
import pickle
import scipy
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, dense_to_sparse, to_dense_adj, is_undirected, remove_self_loops, degree, to_scipy_sparse_matrix, from_scipy_sparse_matrix
from deeprobust.graph.defense import GCN
import scipy.sparse as sp
import yaml
from yaml import SafeLoader
from torch_scatter import scatter
import os

def calculate_classification_metrics(logits, targets_onehot, num_classes=None, average='macro'):
    """
    计算多分类任务的精确率、召回率和F1分数
    
    参数:
        logits: 模型输出的logits，形状为 (batch_size, num_classes)
        targets_onehot: 真实标签的one-hot表示，形状为 (batch_size, num_classes)
        num_classes: 类别数量，默认为None(自动从数据中推断)
        average: 计算方式，可选 'macro'、'micro'、'weighted' 或 None(返回各类别指标)
    
    返回:
        precision: 精确率
        recall: 召回率
        f1_score: F1分数
    """
    # 将logits转换为预测类别索引
    preds_indices = logits.argmax(dim=1)
    
    # 将one-hot标签转换为类别索引
    if targets_onehot.dim() == 1:
        targets_indices = targets_onehot
    else:
        targets_indices = targets_onehot.argmax(dim=1)
    
    # 自动推断类别数
    if num_classes is None:
        num_classes = max(logits.shape[1], targets_indices.max().item() + 1)
    
    # 计算混淆矩阵元素
    tp = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    fp = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    fn = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    
    for c in range(num_classes):
        # 当前类别的预测和真实标签
        preds_c = (preds_indices == c)
        targets_c = (targets_indices == c)
        
        tp[c] = (preds_c & targets_c).sum()
        fp[c] = (preds_c & (~targets_c)).sum()
        fn[c] = ((~preds_c) & targets_c).sum()
    
    # 计算各类别的精确率、召回率和F1分数
    precision_per_class = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    recall_per_class = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    f1_per_class = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    
    # 避免除零错误
    mask = (tp + fp) > 0
    precision_per_class[mask] = tp[mask] / (tp[mask] + fp[mask])
    
    mask = (tp + fn) > 0
    recall_per_class[mask] = tp[mask] / (tp[mask] + fn[mask])
    
    mask = (precision_per_class + recall_per_class) > 0
    f1_per_class[mask] = 2 * (precision_per_class[mask] * recall_per_class[mask]) / (precision_per_class[mask] + recall_per_class[mask])
    
    # 根据average参数聚合结果
    if average == 'macro':
        # 计算宏平均（各类别平等权重）
        precision = precision_per_class.mean()
        recall = recall_per_class.mean()
        f1_score = f1_per_class.mean()
    elif average == 'micro':
        # 计算微平均（全局TP、FP、FN）
        total_tp = tp.sum()
        total_fp = fp.sum()
        total_fn = fn.sum()
        
        if total_tp + total_fp > 0:
            precision = total_tp / (total_tp + total_fp)
        else:
            precision = torch.tensor(0.0, device=logits.device)
        
        if total_tp + total_fn > 0:
            recall = total_tp / (total_tp + total_fn)
        else:
            recall = torch.tensor(0.0, device=logits.device)
        
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = torch.tensor(0.0, device=logits.device)
    elif average == 'weighted':
        # 计算加权平均（按类别样本数加权）
        class_weights = torch.bincount(targets_indices, minlength=num_classes).float()
        total_samples = class_weights.sum()
        
        if total_samples > 0:
            weights = class_weights / total_samples
            precision = (precision_per_class * weights).sum()
            recall = (recall_per_class * weights).sum()
            f1_score = (f1_per_class * weights).sum()
        else:
            precision = torch.tensor(0.0, device=logits.device)
            recall = torch.tensor(0.0, device=logits.device)
            f1_score = torch.tensor(0.0, device=logits.device)
    else:
        # 返回各类别的指标
        precision = precision_per_class
        recall = recall_per_class
        f1_score = f1_per_class
    
    return precision, recall, f1_score

# datapath = './{}}/{}/g_{}_l_{}_f_{}/'.format(noise_type,dataset, graph_noise, l,feature_noise)
# datapath = './pro/{}/g_{}_l_{}/'.format(dataset, graph_noise, l)
class pringle(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.datapath = '../pythonProject1/{}/{}/g_{}_l_{}_f_{}/'.format(args.noise_type_new,args.dataset_new, args.g, args.l,args.f)
    
    def training(self):
        device = f'cuda:{self.args.device}'
        
        self.train_result, self.val_result, self.test_result = defaultdict(list), defaultdict(list), defaultdict(list)
        self.pre_result, self.rec_result, self.f1_result = defaultdict(list), defaultdict(list), defaultdict(list)
        pred_list=[]
        for seed in range(self.args.seed_n):
            self.seed = seed
            set_everything(seed)
            data = self.data.clone()
            data = data.to(device)
            model = FDGN(self.args, device)
            model.fit(data.x, data.edge_index, data.y, data.train_mask, data.val_mask, data.test_mask)


            # with open(self.datapath + '{}_train_index.pkl'.format(self.args.dataset_new), 'rb') as f:
            #     train_idx = pickle.load(f, encoding='latin1')
            # with open(self.datapath + '{}_val_index.pkl'.format(self.args.dataset_new), 'rb') as f:
            #     val_idx = pickle.load(f, encoding='latin1')
            # with open(self.datapath + '{}_test_index.pkl'.format(self.args.dataset_new), 'rb') as f:
            #     test_idx = pickle.load(f, encoding='latin1')
            # with open(self.datapath + '{}_train_label.pkl'.format(self.args.dataset_new, self.args.l), 'rb') as f:
            #     true_train_label = pickle.load(f, encoding='latin1')
            # with open(self.datapath + '{}_train_label_new_{}.pkl'.format(self.args.dataset_new, self.args.l), 'rb') as f:
            #     train_label = pickle.load(f, encoding='latin1')
            # with open(self.datapath + '{}_val_label.pkl'.format(self.args.dataset_new), 'rb') as f:
            #     val_label = pickle.load(f, encoding='latin1')
            # with open(self.datapath + '{}_test_label.pkl'.format(self.args.dataset_new), 'rb') as f:
            #     test_label = pickle.load(f, encoding='latin1')
            # with open(self.datapath + '{}_features.pkl'.format(self.args.dataset_new), 'rb') as f:
            #     features = pickle.load(f, encoding='latin1')
            # adj = scipy.sparse.load_npz(self.datapath + '{}_mod_adj_add_{}.npz'.format(self.args.dataset_new, self.args.g))
            # with open(self.datapath + 'nclass.pkl', 'rb') as f:
            #     num_classes = pickle.load(f, encoding='latin1')
            # with open(self.datapath + '{}_all_label.pkl'.format(self.args.dataset_new), 'rb') as f:
            #     labels = pickle.load(f, encoding='latin1')

            # train_mask = index_to_mask(torch.from_numpy(train_idx.detach().numpy()), features.size(0))
            # val_mask = index_to_mask(torch.from_numpy(val_idx.detach().numpy()), features.size(0))
            # test_mask = index_to_mask(torch.from_numpy(test_idx.detach().numpy()), features.size(0))
            # edge_index = dense_to_sparse(torch.from_numpy(adj.toarray()))[0].long()
            # train_label = train_label.values.reshape(-1)
            # train_label = torch.from_numpy(train_label)
            # train_label = torch.tensor(train_label, dtype=torch.long, device='cuda:0')
            # edge_index = to_undirected(edge_index)
            # features = torch.tensor(features, device='cuda:0')
            # edge_index = torch.tensor(edge_index,  device='cuda:0')
            # labels = torch.tensor(labels,  device='cuda:0')
            # train_mask = torch.tensor(train_mask, device='cuda:0')
            # val_mask = torch.tensor(val_mask,  device='cuda:0')
            # test_mask = torch.tensor(test_mask, device='cuda:0')
            # labels_c=labels.clone()
            # labels[train_idx] = train_label
            # num=0
            # for i in range(2708):
            #     if labels_c[i] != labels[i]:
            #         num += 1
            # actual_noise = num/140
            # print('actual_l:', actual_noise)

            # model.fit(features,edge_index,labels,train_mask,val_mask,test_mask)
            print("===================")
            # intime=[]
            # for i in range(2000):
            test_acc, pred,pre,rec,f1,itime = model.test(data.test_mask)
            #     print(i)
            #     intime.append(itime)
            # print("final_time:{:.4f}".format(np.mean(np.array(intime))),"{:.4f}".format(np.std(np.array(intime))))
            pred_list.append(pred)

            self.train_result[f'result'].append(0)
            self.val_result[f'result'].append(0)
            self.test_result[f'result'].append(test_acc)
            self.pre_result[f'result'].append(pre)
            self.rec_result[f'result'].append(rec)
            self.f1_result[f'result'].append(f1)
        os.makedirs('./dataset_pro', exist_ok=True)
        torch.save(pred_list, os.path.join('./dataset_pro/', f'test_{self.args.noise_rate}.pt'))
        self.summary_result()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
import utils

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, self_loop=True, device=None):

        super(GCN, self).__init__()

        self.device = device
        self.gc1 = GCNConv(nfeat, nhid, bias=True, add_self_loops=self_loop)
        self.gc2 = GCNConv(nhid, nclass, bias=True, add_self_loops=self_loop)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x1 = F.relu(self.gc1(x, edge_index, edge_weight))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = self.gc2(x1, edge_index, edge_weight)
        return x1

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()


import numpy as np
from numpy.testing import assert_array_almost_equal
import torch

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))


def accuracy(output, labels):
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.utils as utils
import scipy.sparse as sp

def kl_loss_compute(pred, soft_targets, reduce=True, tempature=1):
    pred = pred / tempature
    soft_targets = soft_targets / tempature
    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduce=False)
    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

class NeighborConsistency(nn.Module):
    def __init__(self,device='cuda'):
        super(NeighborConsistency, self).__init__()
        self.device = device

    def neighbor_cons(self,y_1,edge_index,edge_weight):
        weighted_adj = utils.to_scipy_sparse_matrix(edge_index, edge_weight.detach())
        colsum = np.array(weighted_adj.sum(0))
        r_inv = np.power(colsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        norm_adj = weighted_adj.dot(r_mat_inv)

        norm_idx, norm_weight = utils.from_scipy_sparse_matrix(norm_adj)
        norm_idx, norm_weight = norm_idx.to(self.device), norm_weight.to(self.device)

        ncr = (norm_weight*kl_loss_compute(y_1[norm_idx[1]], y_1[norm_idx[0]].detach())).sum()
        ncr = ncr/y_1.shape[0]
        return ncr

    def forward(self,y_1,edge_index,edge_weight):
        neighbor_kl_loss = self.neighbor_cons(y_1, edge_index, edge_weight)
        return neighbor_kl_loss


class FDGN(nn.Module):
    def __init__(self, args, device):
        super(FDGN, self).__init__()
        self.device = device
        self.args = args
        self.best_acc_pred_val = 0
        self.nc_reg = NeighborConsistency()
        self.label_smoothing = None
        self.datapath = '../pythonProject1/{}/{}/g_{}_l_{}_f_{}/'.format(args.noise_type_new,args.dataset_new, args.g, args.l,args.f)

    def fit(self, features, edge_index, labels, train_mask, val_mask, test_mask):
        args = self.args
        self.edge_index = edge_index.clone()
        self.features = features.clone()
        self.labels = labels.clone()
        self.idx_train = train_mask.nonzero().flatten()

        self.p_theta2 = eps_Decoder(args, features, labels).to(self.device)
        self.p_theta3 = GCN(nfeat=features.shape[1]+labels.max().item() + 1,
                                  nhid=self.args.hidden,
                                  nclass=labels.max().item() + 1,
                                  dropout=self.args.dropout, device=self.device).to(self.device)

        self.q_phi1 = q_phi1(features, features.shape[1], args, device=self.device).to(self.device)
        self.q_phi2 = eps_Encoder(args, features, labels).to(self.device)
        self.q_phi3 = GCN(nfeat=features.shape[1],
                                  nhid=self.args.hidden,
                                  nclass=labels.max().item() + 1,
                                  dropout=self.args.dropout, device=self.device).to(self.device)
        self.features_sim = F.normalize(features, dim=1, p=2).mm(F.normalize(features, dim=1, p=2).T).fill_diagonal_(0.0)
        # Prior of Z_A
        if self.args.K > 0:
            if self.args.hop == 0:
                feats = features.clone()
            elif self.args.hop > 0:
                tmp_edge = tu.add_self_loops(edge_index=edge_index, num_nodes=features.shape[0])[0]
                feats = scatter(features[tmp_edge[1]], tmp_edge[0], dim=0, dim_size=features.size(0), reduce='mean')
            sim = F.normalize(feats, dim=1, p=2).mm(F.normalize(feats, dim=1, p=2).T).fill_diagonal_(0.0)
            dst = sim.topk(self.args.K, 1)[1].to(self.device)
            src = torch.arange(sim.size(0)).unsqueeze(1).expand_as(dst).to(self.device)
            knn_edge = torch.stack([src.reshape(-1), dst.reshape(-1)])
            self.pred_edge_index = tu.to_undirected(knn_edge).to(self.device)
            # self.pred_edge_index =edge_index.clone()
        else:
            self.pred_edge_index = edge_index.clone()
        self.pred_edge_index = tu.coalesce(torch.cat([edge_index.to(self.device), self.pred_edge_index.to(self.device)], dim=1))
        self.optimizer = optim.Adam(list(self.q_phi1.parameters()) + list(self.q_phi2.parameters()) + list(self.q_phi3.parameters()) + list(self.p_theta2.parameters())  + list(self.p_theta3.parameters()) ,
                                    lr=args.lr, weight_decay=args.weight_decay)
        cnt_wait=0
        time_list=[]
        for epoch in range(args.epochs):
            val_acc,t = self.train(epoch, features, edge_index, train_mask, val_mask, test_mask)
            time_list.append(t)
            if val_acc >= self.best_acc_pred_val:
                cnt_wait = 0
            else:
                cnt_wait += 1
            
            if cnt_wait == self.args.patience:
                print('Early stopping!')
                break           
        print("Optimization Finished!")
        
      
        print("picking the best model according to validation performance")
        print(np.mean(time_list),np.std(time_list))
        self.q_phi3.load_state_dict(self.predictor_model_weigths)


    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def filter_noisy_labels(self, y_1, edge_index, edge_weight, train_mask):
        is_dst_train = train_mask[edge_index[1]]
        target_edge_index = edge_index[:, is_dst_train]
        target_edge_weight = edge_weight[is_dst_train]
        src_y, dst_y = y_1[target_edge_index]
        is_same_y = (src_y == dst_y).float() * target_edge_weight
        knn_vote = scatter(is_same_y, target_edge_index[1], dim=0, dim_size=train_mask.size(0), reduce='mean')
        # idx_train = knn_vote.topk(int(train_mask.sum(0) * self.args.sel_prop))[1].long()
        return knn_vote
        

    def train(self, epoch, features, edge_index, train_mask, val_mask, test_mask):
        pred_mask = train_mask.clone().nonzero().flatten()
        t = time.time()
        args = self.args
        self.q_phi3.train()
        self.optimizer.zero_grad()
        representations, rec_loss, pos_w = self.q_phi1(edge_index, features, self.features_sim, self.label_smoothing)
        predictor_weights = self.q_phi1.get_estimated_weigths(self.pred_edge_index, representations)
        edge_remain_idx = torch.where(predictor_weights!=0)[0].detach()
        predictor_weights = predictor_weights[edge_remain_idx]
        pred_edge_index = self.pred_edge_index[:,edge_remain_idx]

        output_z_y = self.q_phi3(features, pred_edge_index, predictor_weights)
        pred = F.softmax(output_z_y, dim=1)
        # if epoch >= 0:
        #     with torch.no_grad():pro_pair
        #         score = self.filter_noisy_labels(pred.argmax(1), pred_edge_index, predictor_weights, train_mask)
        z_y = F.gumbel_softmax(output_z_y)
        KLD_Z_Y = self.nc_reg(output_z_y, pred_edge_index, predictor_weights)
        
        mu_eps_x, logvar_eps_x = self.q_phi2(features, z_y)
        KLD_Z_X = kld_gaussian_prior(mu_eps_x, logvar_eps_x)
        eps_x = self.q_phi2._z_x_reparameterize(mu_eps_x, logvar_eps_x)
        xhat = self.p_theta2(eps_x, z_y)

        if self.args.dataset not in ['pubmed', 'wikics']:
            recon_Xhat_X_loss = F.binary_cross_entropy_with_logits(xhat, features, reduction='mean')
        else:
            recon_Xhat_X_loss = F.mse_loss(xhat, features, reduction='mean')
            
        output_y = self.p_theta3(torch.cat((features, z_y), dim=1), edge_index, None)


        # loss_gcn_infer_clean_y = torch.sum(F.cross_entropy(output_z_y[train_mask],self.labels[train_mask], reduction='none') * score[train_mask].softmax(0))
        loss_gcn_infer_clean_y = F.cross_entropy(output_z_y[train_mask],self.labels[train_mask], reduction='mean')
        loss_gcn_infer_noisy_y = F.cross_entropy(output_y[train_mask],self.labels[train_mask])

        # total_loss = loss_gcn_infer_clean_y + self.args.alpha * rec_loss + self.args.beta*KLD_Z_Y+(loss_gcn_infer_noisy_y+KLD_Z_X+recon_Xhat_X_loss)*0.001
        total_loss = loss_gcn_infer_clean_y + self.args.alpha * rec_loss+(KLD_Z_X+loss_gcn_infer_noisy_y+recon_Xhat_X_loss)*0.001+ self.args.beta*KLD_Z_Y
        total_loss.backward()
        self.optimizer.step()

        self.q_phi3.eval()
        output0 = self.q_phi3(features, pred_edge_index, predictor_weights)
        acc_pred_train0 = accuracy(output0[train_mask], self.labels[train_mask])
        acc_pred_val0 = accuracy(output0[val_mask], self.labels[val_mask])
        acc_pred_test0 = accuracy(output0[test_mask], self.labels[test_mask])

        if acc_pred_val0 >= self.best_acc_pred_val:
            print(1)
            self.best_acc_pred_val = acc_pred_val0
            self.best_pred_graph = predictor_weights.detach()
            self.best_edge_idx = pred_edge_index.detach()
            self.predictor_model_weigths = deepcopy(self.q_phi3.state_dict())
        ti=time.time() - t
        print('Epoch: {:04d}'.format(epoch+1),
                      'acc_train: {:.4f}'.format(acc_pred_train0.item()),
                      'acc_val: {:.4f}'.format(acc_pred_val0.item()),
                      'acc_test: {:.4f}'.format(acc_pred_test0.item()),
                      'time: {:.4f}s'.format(ti))

        if epoch <= 30:
            with torch.no_grad():
                tmp_edge = edge_index[:, edge_index[0] < edge_index[1]]
                if epoch == 0:
                    self.phat = torch.ones_like(tmp_edge[0]).float()
                phat = F.relu(F.cosine_similarity(representations[tmp_edge[0]], representations[tmp_edge[1]], dim=1))
                self.phat = self.args.decay * self.phat + (1-self.args.decay) * phat
                min_value, max_value = self.phat.min(), self.phat.max()
                normalized_tensor = (self.phat - min_value) / (max_value - min_value)
                self.label_smoothing = 0.1 * normalized_tensor + 0.9
                
        return acc_pred_val0,ti

    def test(self, idx_test):

        with open(self.datapath + '{}_features.pkl'.format(self.args.dataset_new), 'rb') as f:
            features = pickle.load(f, encoding='latin1')
        adj = scipy.sparse.load_npz(self.datapath + '{}_mod_adj_add_{}.npz'.format(self.args.dataset_new, self.args.g))
        with open(self.datapath + 'nclass.pkl', 'rb') as f:
            num_classes = pickle.load(f, encoding='latin1')
        with open(self.datapath + '{}_all_label.pkl'.format(self.args.dataset_new), 'rb') as f:
            labels = pickle.load(f, encoding='latin1')
        # features = self.features
        features = torch.tensor(features, device='cuda:0')
        labels = torch.tensor(labels, device='cuda:0')
        
        self.q_phi3.eval()
        estimated_weights = self.best_pred_graph
        pred_edge_index = self.best_edge_idx
        t1=time.time()
        output0 = self.q_phi3(features, pred_edge_index, estimated_weights)
        t2=time.time()
        acc_pred_test0 = accuracy(output0[idx_test], labels[idx_test]).item()
        pre,rec,f1=calculate_classification_metrics(output0[idx_test], labels[idx_test])
        pre=pre.item()
        rec=rec.item()
        f1=f1.item()
        print("Test Accuray: #1 = %f"%(acc_pred_test0))
        return acc_pred_test0, output0.detach(),pre,rec,f1,t2-t1

class q_phi1(nn.Module):

    def __init__(self, features, nfea, args, device='cuda'):
        super(q_phi1, self).__init__()
        self.estimator = GCN(nfea, args.edge_hidden, args.edge_hidden, dropout=0.0, device=device)
        self.device = device
        self.args = args

    def forward(self, edge_index, features, sim, label_smoothing):
        representations = self.estimator(features, edge_index, \
                                         torch.ones([edge_index.shape[1]]).to(self.device).float())
        representations =F.normalize(representations,dim=-1)
        rec_loss, pos_w = self.reconstruct_loss(edge_index, representations, sim, label_smoothing)
        return representations, rec_loss, pos_w

    def get_estimated_weigths(self, edge_index, representations):
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0, x1), dim=1)
        estimated_weights = F.relu(output)
        if estimated_weights.shape[0] != 0:
            estimated_weights = estimated_weights.masked_fill(estimated_weights < self.args.tau, 0.0)
            
        return estimated_weights

    def reconstruct_loss(self, edge_index, representations, sim, label_smoothing):
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index, num_nodes=num_nodes,
                                        num_neg_samples=edge_index.size(1))
        randn = randn[:, randn[0] < randn[1]]

        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg_w = F.relu(torch.sum(torch.mul(neg0, neg1), dim=1))

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos_w = F.relu(torch.sum(torch.mul(pos0, pos1), dim=1))

        pos_fsim = sim[edge_index[0], edge_index[1]]
        
        pos = pos_fsim * self.args.theta + pos_w * (1-self.args.theta)
        

        if label_smoothing is None:
            rec_loss = (F.mse_loss(neg_w, torch.zeros_like(neg_w), reduction='sum') \
                        + F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) \
                       * num_nodes / (randn.shape[1] + edge_index.shape[1])

        else:
            rec_loss = (F.mse_loss(neg_w, torch.zeros_like(neg_w), reduction='sum') \
                        + F.mse_loss(pos, label_smoothing, reduction='sum')) \
                       * num_nodes / (randn.shape[1] + edge_index.shape[1])

        return rec_loss, pos_w



class eps_Encoder(nn.Module):
    def __init__(self, args, features, labels):
        super(eps_Encoder, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(features.size(1) + labels.unique().size(0), args.edge_hidden)
        self.fc2 = nn.Linear(args.edge_hidden, args.edge_hidden)
        self.fc_mu = nn.Linear(args.edge_hidden, 16)
        self.fc_logvar = nn.Linear(args.edge_hidden, 16)

    def forward(self, x, y_hat):
        out = torch.cat((x, y_hat), dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        mu = F.elu(self.fc_mu(out))
        logvar = F.elu(self.fc_logvar(out))
        return mu, logvar

    def _z_x_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

class eps_Decoder(nn.Module):
    def __init__(self, args, features, labels):
        super(eps_Decoder, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(16 + labels.unique().size(0), args.edge_hidden)
        self.fc2 = nn.Linear(args.edge_hidden, args.edge_hidden)
        self.recon = nn.Linear(args.edge_hidden, features.size(1))
        
    def forward(self, z_x, y_hat):
        out = torch.cat((z_x, y_hat), dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        x_hat = F.elu(self.recon(out))
        return x_hat
    
def kld_gaussian_prior(mu, log_var):
    return -torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()/2
