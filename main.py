# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3950
import argparse
from cProfile import label

import torch
from models.model_unnamed5 import unnamed_model5
from models.model_unnamed4 import unnamed_model4
from models.model_unnamed3 import unnamed_model3
from utils.load_noisydata import load_noisydata
import os
from models.Semi_RNCGLN import *
import numpy as np
import torch_geometric.utils as tu
from torch_scatter import scatter
import pickle
from scipy.sparse import coo_matrix
from models.model_unnamed4 import to_edge_index,to_adj
import torch.nn.functional as F
import warnings
import pandas as pd
from data import get_data
warnings.filterwarnings("ignore")

import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:19000"

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser(
    description="PyTorch implementation of model")
parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='dataset used.Cora,CiteSeer,Photo,PubMed,ogbn-arxiv,ogbn-products')
parser.add_argument('--noise_type', type=str, default='uniform', help='uniform,pair_noise,class_confusion,feature_based')
parser.add_argument('--teacher_student', type=str, default='teacher', help='teacher,student')
parser.add_argument('--epochs', type=int, default=5000,
                    help='number of epochs to train ')
parser.add_argument('--graph_noise', type=float, default=0.1,
                    help='rate of the graph_noise')
parser.add_argument('--label_noise', type=float, default=0.1,
                    help='rate of the label_noise')
parser.add_argument('--feature_noise', type=float, default=0.1, 
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
parser.add_argument("--K", type=int, default=200)
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
parser.add_argument('--num_parts', type=int, default=4, help='number of METIS partitions')
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
    pre_list=[]
    rec_list=[]
    intime_list = []
    for seed in range(2020, 2024):
        print('seed:{}'.format(seed))
        
        part_test_accs = []

        part_test_counts = []
        part_pres = []
        part_recs = []
        part_f1s = []
        part_times = []

        for part_id in range(args.num_parts):
            actual_part_id = part_id if args.num_parts > 1 else None
            args.part_id = actual_part_id
            if args.num_parts > 1:
                print(f'--- Training on Partition {part_id}/{args.num_parts} ---')

            (args.train_index, args.val_index, args.test_index, args.train_label, args.val_label, args.test_label,
             args.features, args.adj, args.true_train_label, args.num_classes, args.labels) = load_noisydata(
                noise_type=args.noise_type,
                dataset=args.dataset,
                graph_noise=args.graph_noise,
                label_noise=args.label_noise,
                feature_noise=args.feature_noise,
                part_id=actual_part_id
            )

            args.train_label = args.train_label.values.reshape(-1)
            args.train_label = torch.from_numpy(args.train_label)
            args.train_label = torch.tensor(args.train_label, dtype=torch.long, device='cuda:0')

            args.train_index = torch.tensor(args.train_index, dtype=torch.long, device='cuda:0')
            args.train_index_clean = args.train_index.clone()
            args.val_index = torch.tensor(args.val_index, dtype=torch.long, device='cuda:0')
            args.test_index = torch.tensor(args.test_index, dtype=torch.long, device='cuda:0')

            print(f"Partition {part_id} Statistics:")
            print(f"  Total Nodes: {args.features.shape[0]}")
            print(f"  Train Nodes: {args.train_index.size(0)}")
            print(f"  Val Nodes:   {args.val_index.size(0)}")
            print(f"  Test Nodes:  {args.test_index.size(0)}")

            args.ft_size = args.features.shape[1]
            args.features = torch.tensor(args.features, dtype=torch.float32, device='cuda:0')
            args.labels = torch.tensor(args.labels, dtype=torch.long, device='cuda:0')
            args.input_size = args.features[args.train_index].shape[1]
            args.features1=args.features.clone()
            args.features2 = args.features.clone()
            args.adj = args.adj.tocoo()
            args.num_edge = args.adj.nnz
            args.labels_c = args.labels.clone()
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

            args.edge_index, args.edge_weight = to_edge_index(args.adj)

            try:
                (args.clean_train_index, args.clean_val_index, args.clean_test_index, args.clean_train_label, args.clean_val_label, args.clean_test_label,
                 args.clean_features, args.clean_adj, args.clean_true_train_label, args.clean_num_classes, args.clean_labels) = load_noisydata(
                    noise_type="uniform",
                    dataset=args.dataset,
                    graph_noise=0.0,
                    label_noise=0.0,
                    feature_noise=0.0,
                    part_id=actual_part_id
                )
                args.clean_labels=torch.tensor(args.clean_labels, device='cuda:0')
                args.clean_features = torch.tensor(args.clean_features, dtype=torch.float32, device='cuda:0')
                args.clean_adj = args.clean_adj.tocoo()
                args.clean_edge_index, args.clean_edge_weight = to_edge_index(args.clean_adj)
                
                if args.features1.size(0) == args.clean_features.size(0):
                    indices=[]
                    for i in range(args.features1.size(0)):
                        if torch.equal(args.features1[i],args.clean_features[i]):
                            indices.append(i)
                    filepath1 = r'./feature_truth/feature_clean.csv'
                    indices=np.array(indices)
                    df1= pd.DataFrame(indices)
                    df1.to_csv(filepath1, index=False)
            except Exception as e:
                if args.num_parts == 1:
                    print(f"Warning: Could not load clean data: {e}")

            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
                
            if args.teacher_student=='student':
                path='./teacher_models/'+args.noise_type+'/'+args.dataset
                part_suffix = f"_part{args.part_id}" if args.part_id is not None else ""
                try:
                    with open(path+'/g_{}_l_{}_f_{}/soft_label2020{}.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise, part_suffix), 'rb') as f:
                        args.soft_label1 = pickle.load(f)
                    with open(path+'/g_{}_l_{}_f_{}/soft_label2021{}.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise, part_suffix), 'rb') as f:
                        args.soft_label2 = pickle.load(f)
                    with open(path+'/g_{}_l_{}_f_{}/soft_label2022{}.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise, part_suffix), 'rb') as f:
                        args.soft_label3 = pickle.load(f)
                    with open(path+'/g_{}_l_{}_f_{}/soft_label2023{}.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise, part_suffix), 'rb') as f:
                        args.soft_label4 = pickle.load(f)
                except:
                    pass
            
            model = unnamed_model4(args, args.input_size, args.hidden_size, args.output_size, args.num_classes)
            model.fit()
            
            test_acc, pred, pre, rec, f1, itime = model.test(args.test_index, seed, args.noise_type)
            
            part_test_accs.append(test_acc)
            part_test_counts.append(args.test_index.size(0))
            part_pres.append(pre.item())
            part_recs.append(rec.item())
            part_f1s.append(f1.item())
            part_times.append(itime)

        total_test_nodes = sum(part_test_counts)
        if total_test_nodes > 0:
            seed_acc = sum(acc * count for acc, count in zip(part_test_accs, part_test_counts)) / total_test_nodes
            seed_pre = sum(pre * count for pre, count in zip(part_pres, part_test_counts)) / total_test_nodes
            seed_rec = sum(rec * count for rec, count in zip(part_recs, part_test_counts)) / total_test_nodes
            seed_f1 = sum(f1 * count for f1, count in zip(part_f1s, part_test_counts)) / total_test_nodes
            seed_time = np.mean(part_times)
        else:
            seed_acc = seed_pre = seed_rec = seed_f1 = seed_time = 0

        result_path = f'./summary_result/{args.noise_type}_{args.dataset}_{args.graph_noise}_{args.label_noise}_{args.feature_noise}.txt'
        mode = 'a' if os.path.exists(result_path) else 'w'
        with open(result_path, mode) as f:
            f.write(str(seed) + ":")
            f.write(str(seed_acc))
            f.write(f'\n')
            
        acc_list.append(seed_acc*100)
        pre_list.append(seed_pre*100)
        rec_list.append(seed_rec*100)
        f1_list.append(seed_f1)
        intime_list.append(seed_time)

    result_path = f'./summary_result/{args.noise_type}_{args.dataset}_{args.graph_noise}_{args.label_noise}_{args.feature_noise}_final.txt'
    with open(result_path, 'w') as f:
        f.write("final_acc:{:.4f},std:{:.4f}".format(np.mean(np.array(acc_list)),np.std(np.array(acc_list))))
    print("final_acc:{:.2f}".format(np.mean(np.array(acc_list))),"{:.2f}".format(np.std(np.array(acc_list))))
    print("final_pre:{:.2f}".format(np.mean(np.array(pre_list))),"{:.2f}".format(np.std(np.array(pre_list))))
    print("final_rec:{:.2f}".format(np.mean(np.array(rec_list))),"{:.2f}".format(np.std(np.array(rec_list))))
    print("final_f1:{:.2f}".format(np.mean(np.array(f1_list))),"{:.2f}".format(np.std(np.array(f1_list))))
    print("final_time:{:.4f}".format(np.mean(np.array(intime_list))),"{:.4f}".format(np.std(np.array(intime_list))))

if __name__ == "__main__":
    main()