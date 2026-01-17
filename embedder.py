import numpy as np
from utils_p import get_data, set_cuda_device, config2string, ensure_dir, to_numpy, get_2hop_edge_index
import os
from copy import deepcopy
from datetime import datetime
from collections import defaultdict
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_undirected

class embedder:
    def __init__(self, args):
        print('===',args.dataset, '===') 
        self.args = args
        self.device = f'cuda:{args.device}'
        set_cuda_device(args.device)

        self.data_home = f'./dataset/'
        self.data = get_data(self.data_home, args.dataset, args.noise_type, args.noise_rate, args.label_rate)
        self.clean = get_data(self.data_home, args.dataset, 'clean', args.noise_rate, args.label_rate)
        
        # save results
        self.config_str = config2string(args)
        self.train_result, self.val_result, self.test_result = defaultdict(list), defaultdict(list), defaultdict(list)

        # basic statistics
        self.args.in_dim = self.data.x.shape[1]
        self.args.n_class = self.data.y.unique().size(0)
        self.args.n_node = self.data.x.shape[0]

        ensure_dir(f'{args.save_dir}/saved_model/')
        ensure_dir(f'{args.save_dir}/summary_result/{args.embedder}/bypass/')
        
    def summary_result(self):
        
        assert self.train_result.keys() == self.val_result.keys() and self.train_result.keys() == self.test_result.keys()

        key = list(self.train_result.keys())
        result_path = f'{self.args.save_dir}/summary_result/{self.args.embedder}/{self.args.dataset}_{self.args.noise_rate}_{self.args.noise_type}.txt'
        mode = 'a' if os.path.exists(result_path) else 'w'
        with open(result_path, mode) as f:
            f.write(self.config_str)
            f.write(f'\n')
            for k in key:
                f.write(f'====={k}=====')
                f.write(f'\n')
                train_mean, train_std = np.mean(self.train_result[k]), np.std(self.train_result[k])
                val_mean, val_std = np.mean(self.val_result[k]), np.std(self.val_result[k])
                test_mean, test_std = np.mean(self.test_result[k]), np.std(self.test_result[k])
                test_mean_pre, test_std_pre = np.mean(self.pre_result[k]), np.std(self.pre_result[k])
                test_mean_rec, test_std_rec = np.mean(self.rec_result[k]), np.std(self.rec_result[k])
                test_mean_f1, test_std_f1 = np.mean(self.f1_result[k]), np.std(self.f1_result[k])
                f.write(f'Train Acc: {train_mean*100:.2f}±{train_std*100:.2f}, Val Acc: {val_mean*100:.2f}±{val_std*100:.2f}, Test Acc: {test_mean*100:.2f}±{test_std*100:.2f}')
                f.write(f'\n')
                f.write(f'Seed List: {[np.round(a*100,2) for a in self.test_result[k]]}')
                f.write(f'\n')
                f.write(f'='*40)
                f.write(f'\n')
            print(f'Train Acc: {train_mean:.4f}±{train_std*100:.2f}, Val Acc: {val_mean*100:.2f}±{val_std*100:.2f}, Test Acc: {test_mean*100:.2f}±{test_std*100:.2f}')
            print(f'Test Acc: {test_mean*100:.2f} {test_std*100:.2f}')
            print(f'Test pre: {test_mean_pre*100:.2f} {test_std_pre*100:.2f}')
            print(f'Test rec: {test_mean_rec*100:.2f} {test_std_rec*100:.2f}')
            print(f'Test f1: {test_mean_f1:.2f} {test_std_f1:.2f}')