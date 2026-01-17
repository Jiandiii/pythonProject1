# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3950
import argparse
from cProfile import label

import torch
# from models.model_unnamed5 import unnamed_model5
from models.model_unnamed4 import unnamed_model4
from utils.load_noisydata import load_noisydata
import os
from models.Semi_RNCGLN import *
import numpy as np
import torch_geometric.utils as tu
from torch_scatter import scatter
import pickle
from scipy.sparse import coo_matrix
from models.model_unnamed4 import to_edge_index
import torch.nn.functional as F
import warnings
from torch import nn
from models.gnn_models import GAT,GCN
import tqdm 
warnings.filterwarnings("ignore")

import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:19000"

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser(
    description="PyTorch implementation of model")
parser.add_argument('--dataset', type=str, default='Cora', help='dataset used.Cora,CiteSeer,Photo,PubMed')
parser.add_argument('--noise_type', type=str, default='feature_based', help='uniform,pair_noise,class_confusion,feature_based')
parser.add_argument('--teacher_student', type=str, default='teacher', help='teacher,student')
parser.add_argument('--epochs', type=int, default=5000,
                    help='number of epochs to train ')
parser.add_argument('--graph_noise', type=float, default=0.3,
                    help='rate of the graph_noise')
parser.add_argument('--label_noise', type=float, default=0.3,
                    help='rate of the label_noise')
parser.add_argument('--feature_noise', type=float, default=0.3, 
                    help='rate of the feature_noise')
parser.add_argument('--hidden_size', type=float, default=128,
                    help='dimension of hidden representation')
parser.add_argument('--output_size', type=float, default=128,
                    help='dimension of the output')

parser.add_argument('--patience', type=int, default=100, help='early stop.')
parser.add_argument('--warmup_num', type=int, default=50, help='epoch for warm-up.')
parser.add_argument('--lr', type=float, default=0.005, help='learning ratio.')
parser.add_argument('--wd', type=float, default=5e-4, help='weight delay.')
parser.add_argument('--random_aug_feature', type=float, default=0.4, help='dropout in hidden layers.')
parser.add_argument('--Trans_layer_num', type=int, default=2, help='layers number for self-attention.')
parser.add_argument('--trans_dim', type=int, default=128, help='hidden dimension for transformer.')
parser.add_argument('--nheads', type=int, default=8, help='number of heads in self-attention.')
parser.add_argument('--dropout', type=float, default=0.6, help='dropout in self-attention layers.')
parser.add_argument("--K", type=int, default=800)
parser.add_argument("--z_dim", type=int, default=16)
parser.add_argument("--tau", type=int, default=0.05)
parser.add_argument('--theta', type=float, default=0.1)
parser.add_argument("--hop", type=int, default=0)
parser.add_argument('--decay', type=float, default=0.9)
parser.add_argument('--label_smoothing', type=float, default=None)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--p_u', type=float, default=0.8, help='threshold of adding pseudo labels')
parser.add_argument("--n_n", type=int, default=50, help='number of negitive pairs per node')
parser.add_argument("--record", type=bool, default=False, help='')
parser.add_argument("--teacher_record", type=bool, default=True, help='')
parser.add_argument("--plus", type=bool, default=False, help='')
parser.add_argument('--P_sel', type=float, default=0.8, help='ratio to preserve 1-0.9 pseudo labels.')
parser.add_argument('--P_sel_onehot', type=float, default=0.9, help='ratio to preserve 1-0.9 one-hot label.')
parser.add_argument('--lambda_kld', type=float, default=300, help='ratio of kld loss')
parser.add_argument('--ntea', type=int, default=4, help='num of teacher models')
args = parser.parse_args()


def main():
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
    args.features_old_ind1=[]
    args.features_old_ind2=[]
    args.top_coords1=[]
    args.top_coords2=[]

    # n=0
    # for i in range(2708):
    #     n+=np.count_nonzero(args.adj.toarray()[i])
    # print(n)
    args.adj = torch.tensor(args.adj.toarray(), device='cuda:0')
    args.edge_index, args.edge_weight = to_edge_index(args.adj)

    args.num_gradual = 80
    args.exponent = 1
    args.rate_schedule = np.ones(args.epochs) * args.label_noise*2/3
    args.rate_schedule[:args.num_gradual] = np.linspace(0, args.label_noise*2/3 ** args.exponent, args.num_gradual)


    q_phi3 = GCN(nfeat=args.features.shape[1],
                            nhid=args.hidden_size,
                            nclass=args.labels.max().item() + 1,
                            dropout=args.dropout, device=args.device).to(args.device)
    predictor_model_weigths1 = torch.load(f"./saved_model_weights/{args.dataset}.pth")
    q_phi3.load_state_dict(predictor_model_weigths1)
    i_time = []
    for i in tqdm(range(2000)):
        t3 = time()
        out = q_phi3(args.features1.data, args.edge_index, args.edge_weight)
        t4 = time()
        i_time.append(t4 - t3)
    print(f"Average inference time : {np.mean(i_time)}")