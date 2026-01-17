# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3950
import argparse
from cProfile import label

import torch
from models.model_unnamed3 import unnamed_model3
from utils.load_noisydata_gc import load_noisydata_gc
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
warnings.filterwarnings("ignore")

import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:19000"

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser(
    description="PyTorch implementation of model")
parser.add_argument('--dataset', type=str, default='molhiv', help='dataset used.Cora,CiteSeer,Photo,PubMed')
parser.add_argument('--noise_type', type=str, default='uniform_gc', help='uniform_gc,pair_noise_gc,feature_based_gc')
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
parser.add_argument('--lr', type=float, default=0.0005, help='learning ratio.')
parser.add_argument('--wd', type=float, default=5e-4, help='weight delay.')
parser.add_argument('--random_aug_feature', type=float, default=0.4, help='dropout in hidden layers.')
parser.add_argument('--Trans_layer_num', type=int, default=2, help='layers number for self-attention.')
parser.add_argument('--trans_dim', type=int, default=128, help='hidden dimension for transformer.')
parser.add_argument('--nheads', type=int, default=8, help='number of heads in self-attention.')
parser.add_argument('--dropout', type=float, default=0.6, help='dropout in self-attention layers.')
parser.add_argument("--K", type=int, default=50)
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
args = parser.parse_args()


def main():
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    acc_list = []
    training_time_list = []
    ap_list = []
    f1_list = []
    for seed in range(2020, 2024):
        print('seed:{}'.format(seed))
        args.train_batch,args.val_batch,args.test_batch,args.feature_list,args.adj,args.num_classes,args.noise_label_list = load_noisydata_gc(
            noise_type=args.noise_type,
            dataset=args.dataset,
            graph_noise=args.graph_noise,
            label_noise=args.label_noise,
            feature_noise=args.feature_noise
        )
        _,_,_,args.clean_feature_list,args.clena_adj,_,args.clean_label_list= load_noisydata_gc(
            noise_type=args.noise_type,
            dataset=args.dataset,
            graph_noise=0.0,
            label_noise=0.0,
            feature_noise=0.0
        )
        args.edge_index=[]
        args.edge_weight=[]
        for i in range(len(args.adj)):
            args.adj[i]=torch.tensor(args.adj[i],device="cuda:0")
            edge_index, edge_weight = to_edge_index(args.adj[i])
            args.edge_index.append(edge_index)
            args.edge_weight.append(edge_weight)
        args.input_size = args.feature_list[0].shape[1]
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if args.teacher_student=='teacher':
            model = unnamed_model3(args, args.input_size, args.hidden_size, args.output_size, args.num_classes)
        if args.teacher_student=='student':
            path='./teacher_models/'+args.noise_type+'/'+args.dataset
            with open(path+'/g_{}_l_{}_f_{}/soft_label2020.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise), 'rb') as f:
                args.soft_label1 = pickle.load(f)
            with open(path+'/g_{}_l_{}_f_{}/soft_label2021.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise), 'rb') as f:
                args.soft_label2 = pickle.load(f)
            with open(path+'/g_{}_l_{}_f_{}/soft_label2022.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise), 'rb') as f:
                args.soft_label3 = pickle.load(f)
            with open(path+'/g_{}_l_{}_f_{}/soft_label2023.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise), 'rb') as f:
                args.soft_label4 = pickle.load(f)
            model = unnamed_model3(args, args.input_size, args.hidden_size, args.output_size, args.num_classes)

        model.fit()
        print("===================")
        test_acc, pred = model.test(args.test_load,seed,args.noise_type)
        result_path = f'./summary_result/{args.noise_type}_{args.dataset}_{args.graph_noise}_{args.label_noise}_{args.feature_noise}.txt'
        mode = 'a' if os.path.exists(result_path) else 'w'
        with open(result_path, mode) as f:
            f.write(str(seed) + ":")
            f.write(str(test_acc))
            f.write(f'\n')
        acc_list.append(test_acc*100)

    result_path = f'./summary_result/{args.noise_type}_{args.dataset}_{args.graph_noise}_{args.label_noise}_{args.feature_noise}_final.txt'
    with open(result_path, 'w') as f:
        f.write("final_acc:{:.4f},std:{:.4f}".format(np.mean(np.array(acc_list)),np.std(np.array(acc_list))))
    print("final_acc:{:.4f}".format(np.mean(np.array(acc_list))),"std:{:.4f}".format(np.std(np.array(acc_list))))


if __name__ == "__main__":
    main()