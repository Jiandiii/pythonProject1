# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3950
import argparse
from cProfile import label

import torch
from models.model_unnamed6 import unnamed_model6
from models.model_unnamed4 import unnamed_model4
from models.model_unnamed3 import unnamed_model3
from utils.load_noisydata_mao import load_noisydata
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
parser.add_argument('--hidden_size', type=int, default=128,
                    help='dimension of hidden representation')
parser.add_argument('--output_size', type=int, default=128,
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

def get_partition_data(args, part_id):
    actual_part_id = part_id if args.num_parts > 1 else None
    (train_index, val_index, test_index, train_label, val_label, test_label,
     features, adj, true_train_label, num_classes, labels, nodes_all) = load_noisydata(
        noise_type=args.noise_type,
        dataset=args.dataset,
        graph_noise=args.graph_noise,
        label_noise=args.label_noise,
        feature_noise=args.feature_noise,
        part_id=actual_part_id
    )

    train_label = torch.from_numpy(train_label.values.reshape(-1)).long()
    train_index = torch.tensor(train_index, dtype=torch.long)
    val_index = torch.tensor(val_index, dtype=torch.long)
    test_index = torch.tensor(test_index, dtype=torch.long)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    if nodes_all is not None:
        nodes_all = torch.tensor(nodes_all, dtype=torch.long)
    
    labels_noise = labels.clone()
    labels_noise[train_index] = train_label
    
    ones = torch.eye(num_classes)
    targets_oneHot1 = ones.index_select(0, labels_noise)
    targets_c_oneHot = ones.index_select(0, labels)
    
    edge_index, edge_weight = to_edge_index(adj.tocoo())
    
    part_data = {
        'features': features,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'train_index': train_index,
        'val_index': val_index,
        'test_index': test_index,
        'labels': labels,
        'labels_noise': labels_noise,
        'targets_oneHot1': targets_oneHot1,
        'targets_c_oneHot': targets_c_oneHot,
        'num_classes': num_classes,
        'nodes_all': nodes_all
    }
    
    # Load soft labels if student
    if args.teacher_student == 'student':
        path = './teacher_models/' + args.noise_type + '/' + args.dataset
        part_suffix = f"_part{actual_part_id}" if actual_part_id is not None else ""
        try:
            with open(path + f'/g_{args.graph_noise}_l_{args.label_noise}_f_{args.feature_noise}/soft_label2020{part_suffix}.pkl', 'rb') as f:
                part_data['soft_label1'] = pickle.load(f)
            with open(path + f'/g_{args.graph_noise}_l_{args.label_noise}_f_{args.feature_noise}/soft_label2021{part_suffix}.pkl', 'rb') as f:
                part_data['soft_label2'] = pickle.load(f)
            with open(path + f'/g_{args.graph_noise}_l_{args.label_noise}_f_{args.feature_noise}/soft_label2022{part_suffix}.pkl', 'rb') as f:
                part_data['soft_label3'] = pickle.load(f)
            with open(path + f'/g_{args.graph_noise}_l_{args.label_noise}_f_{args.feature_noise}/soft_label2023{part_suffix}.pkl', 'rb') as f:
                part_data['soft_label4'] = pickle.load(f)
        except:
            pass
            
    return part_data

def main():
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    acc_list = []
    pre_list = []
    rec_list = []
    f1_list = []
    intime_list = []

    for seed in range(2020, 2024):
        print(f'seed:{seed}')
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 1. 加载所有分区数据到 CPU
        all_partitions = []
        for part_id in range(args.num_parts):
            print(f"Loading partition {part_id}...")
            part_data = get_partition_data(args, part_id)
            
            # 计算并打印训练集大小和标签噪声比例
            train_idx = part_data['train_index']
            labels_clean = part_data['labels']
            labels_noisy = part_data['labels_noise']
            
            num_train = train_idx.size(0)
            if num_train > 0:
                train_labels_clean = labels_clean[train_idx]
                train_labels_noisy = labels_noisy[train_idx]
                noise_count = (train_labels_clean != train_labels_noisy).sum().item()
                noise_ratio = noise_count / num_train
            else:
                noise_ratio = 0.0
                
            print(f"Partition {part_id}: Train Size = {num_train}, Label Noise Ratio = {noise_ratio:.4f}")
            
            # 计算同质性 (Homophily)
            edge_index = part_data['edge_index']
            labels = part_data['labels']
            if edge_index.numel() > 0:
                row, col = edge_index.cpu()
                labels_cpu = labels.cpu()
                same_label = (labels_cpu[row] == labels_cpu[col]).sum().item()
                homophily = same_label / edge_index.size(1)
            else:
                homophily = 0.0
            print(f"Partition {part_id}: Homophily = {homophily:.4f}")
            
            all_partitions.append(part_data)

        # 计算锚点重叠情况
        all_nodes_sets = [set(p['nodes_all'].tolist()) for p in all_partitions if p['nodes_all'] is not None]
        if len(all_nodes_sets) > 1:
            from collections import Counter
            all_nodes_flat = [node for s in all_nodes_sets for node in s]
            node_counts = Counter(all_nodes_flat)
            anchor_nodes = [node for node, count in node_counts.items() if count > 1]
            print(f"\nTotal Unique Nodes: {len(node_counts)}")
            print(f"Total Anchor Nodes (overlapping): {len(anchor_nodes)}")
            for i, s in enumerate(all_nodes_sets):
                overlap_count = len(s.intersection(anchor_nodes))
                print(f"Partition {i}: Anchor Count = {overlap_count} ({overlap_count/len(s)*100:.2f}% of partition)")
        
        # 2. 初始化模型
        input_size = all_partitions[0]['features'].shape[1]
        num_classes = all_partitions[0]['num_classes']
        
        model = unnamed_model6(args, input_size, args.hidden_size, args.output_size, num_classes, num_domains=args.num_parts)
        
        # 3. 联合训练
        model.fit(all_partitions)

        # 4. 测试每个分区并汇总结果
        part_test_accs = []
        part_test_counts = []
        part_pres = []
        part_recs = []
        part_f1s = []
        part_times = []

        for part_id in range(args.num_parts):
            print(f"Testing on partition {part_id}...")
            # model.test 现在支持传入 partition_data
            test_acc, pred, pre, rec, f1, itime = model.test(
                all_partitions[part_id]['test_index'], 
                seed, 
                args.noise_type,
                partition_data=all_partitions[part_id]
            )
            
            part_test_accs.append(test_acc)
            part_test_counts.append(all_partitions[part_id]['test_index'].size(0))
            part_pres.append(pre.item() if torch.is_tensor(pre) else pre)
            part_recs.append(rec.item() if torch.is_tensor(rec) else rec)
            part_f1s.append(f1.item() if torch.is_tensor(f1) else f1)
            part_times.append(itime)

        # 汇总该 seed 的结果
        total_test_nodes = sum(part_test_counts)
        if total_test_nodes > 0:
            seed_acc = sum(acc * count for acc, count in zip(part_test_accs, part_test_counts)) / total_test_nodes
            seed_pre = sum(pre * count for pre, count in zip(part_pres, part_test_counts)) / total_test_nodes
            seed_rec = sum(rec * count for rec, count in zip(part_recs, part_test_counts)) / total_test_nodes
            seed_f1 = sum(f1 * count for f1, count in zip(part_f1s, part_test_counts)) / total_test_nodes
            seed_time = np.mean(part_times)
        else:
            seed_acc = seed_pre = seed_rec = seed_f1 = seed_time = 0

        acc_list.append(seed_acc * 100)
        pre_list.append(seed_pre * 100)
        rec_list.append(seed_rec * 100)
        f1_list.append(seed_f1)
        intime_list.append(seed_time)

        print(f"Seed {seed} Result: Acc={seed_acc:.4f}, F1={seed_f1:.4f}")

    # 打印最终平均结果
    print("\nFinal Results over all seeds:")
    print("final_acc:{:.2f} ± {:.2f}".format(np.mean(acc_list), np.std(acc_list)))
    print("final_pre:{:.2f} ± {:.2f}".format(np.mean(pre_list), np.std(pre_list)))
    print("final_rec:{:.2f} ± {:.2f}".format(np.mean(rec_list), np.std(rec_list)))
    print("final_f1:{:.2f} ± {:.2f}".format(np.mean(f1_list), np.std(f1_list)))
    print("final_time:{:.4f}".format(np.mean(intime_list)))

if __name__ == "__main__":
    main()
