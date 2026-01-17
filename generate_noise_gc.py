import os
import numpy as np
import pandas as pd
import torch
from utils import Noise_about as Noi_ab
import scipy.sparse as sp
import pickle
from utils import process
from utils import utils
import random

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

if __name__ == '__main__':
    # 导入数据：分隔符为空格
    np.random.seed(2024)
    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    data_list = ['molhiv']
    path_list = ['new_gc','new_pair_gc','feature_based_gc']#'new_gc','new_pair_gc','feature_based_gc','class_confusion_gc'

    for data in data_list:
        label_noise_list = [0.0,0.1,0.3]
        graph_noise_list = [0.0,0.1,0.3]
        feature_noise_list=[0.0,0.1,0.3]


        for label_noise in label_noise_list:
            for graph_noise in graph_noise_list:
                for feature_noise in feature_noise_list:
                    for path in path_list:
                        if path != "new_pair_gc":
                            noise_type='uniform'
                        else:
                            noise_type='pair'
                        print(data, path,label_noise, graph_noise, feature_noise)
                        if not os.path.exists('{}/{}/g_{}_l_{}_f_{}'.format(path,data,graph_noise,label_noise,feature_noise)):
                            os.makedirs('{}/{}/g_{}_l_{}_f_{}'.format(path,data,graph_noise,label_noise,feature_noise))
                        if path !="feature_based_gc":
                            adj, features, labels, idx_train, idx_val, idx_test,node_to_graph,len_train,len_val,len_test = process.load_batch_graph(graph_noise,label_noise,data,type='add')
                        else:
                            adj, features, labels, idx_train, idx_val, idx_test,node_to_graph,len_train,len_val,len_test = process.load_batch_graph(graph_noise,label_noise,data,type='feature_based')
                        if feature_noise !=0:
                            noisy=np.random.choice(range(len(features)), int(len(features) * feature_noise), replace=False)
                            feat_attacked_node = np.sort(noisy)
                            feat_attacked_mask = np.in1d(np.arange(len(features)), feat_attacked_node)

                            #普通噪声
                            if path !="class_confusion_gc":
                                p = torch.from_numpy(features.detach().numpy().mean(1).reshape(-1, 1) * np.ones_like(features.detach().numpy()))
                                if (p > 1 ).any() or (p < 0).any():
                                    perturb = np.random.uniform(features.detach().numpy().min(), features.detach().numpy().max(), size=features.shape) 
                                else:
                                    perturb = p.bernoulli().numpy()
                                perturb = torch.tensor(perturb)
                                features[feat_attacked_mask] = perturb[feat_attacked_mask].to(features.dtype)
                            else:
                                #class_confusion
                                unique_classes = np.unique(labels)

                                for cls in unique_classes:
                                    cls_indices = np.where(labels == cls)[0]
                                    n_cls = len(cls_indices)

                                    # 随机选择当前类别的部分样本（比例由noise_ratio控制）
                                    selected = np.random.choice(cls_indices, size=int(n_cls * feature_noise), replace=False)

                                    # 从其他类别中随机选取替换样本
                                    other_classes = features[labels != cls]
                                    replace_indices = np.random.choice(len(other_classes), size=len(selected), replace=True)

                                    # 执行替换（仅影响选中样本）
                                    features[selected] = other_classes[replace_indices]


                        with open('{}/{}/g_{}_l_{}_f_{}/node_to_graph.pkl'.format(path,data, graph_noise,label_noise,feature_noise),
                                'wb') as f:
                            pickle.dump(node_to_graph, f)
                        with open('{}/{}/g_{}_l_{}_f_{}/len_train.pkl'.format(path,data, graph_noise,label_noise,feature_noise),
                                'wb') as f:
                            pickle.dump(len_train, f)
                        with open('{}/{}/g_{}_l_{}_f_{}/len_val.pkl'.format(path,data, graph_noise,label_noise,feature_noise),
                                'wb') as f:
                            pickle.dump(len_val, f)
                        with open('{}/{}/g_{}_l_{}_f_{}/len_test.pkl'.format(path,data, graph_noise,label_noise,feature_noise),
                                'wb') as f:
                            pickle.dump(len_test, f)

                        root = r'{}/{}/g_{}_l_{}_f_{}/'.format(path,data, graph_noise,label_noise,feature_noise)
                        name = '{}_mod_adj_add_{}.pkl'.format(data, graph_noise)

                        os.makedirs(root, exist_ok=True)

                        with open(root+name.format(path,data, graph_noise,label_noise,feature_noise),
                                'wb') as f:
                            pickle.dump(adj, f)

                        print(idx_train,idx_val,idx_test)
                        
                        # nb_classes = len(set(labels))
                        # labels_flatten= torch.flatten(labels)
                        # features_flatten= features.flatten()
                        nb_classes = int(labels.max() - labels.min()) + 1
                        with open('{}/{}/g_{}_l_{}_f_{}/nclass.pkl'.format(path,data, graph_noise,label_noise,feature_noise),
                                'wb') as f:
                            pickle.dump(nb_classes, f)
                        dim=features.shape[1]
                        with open('{}/{}/g_{}_l_{}_f_{}/dim.pkl'.format(path,data, graph_noise,label_noise,feature_noise),
                                'wb') as f:
                            pickle.dump(dim, f)
                        num_nodes=len(idx_train)
                        all_num_nodes=features.shape[0]
                        if all_num_nodes==idx_train.shape[0]:
                            idx_train_tmp=idx_train
                            idx_val_tmp=idx_val
                            idx_test_tmp=idx_test
                            idx_train=[]
                            idx_val=[]
                            idx_test=[]
                            for i in range(idx_train_tmp.shape[0]):
                                if idx_train_tmp[i]==True:
                                    idx_train.append(i)
                            for i in range(idx_val_tmp.shape[0]):
                                if idx_val_tmp[i]==True:
                                    idx_val.append(i)
                            for i in range(idx_test_tmp.shape[0]):
                                if idx_test_tmp[i]==True:
                                    idx_test.append(i)
                            idx_train=torch.tensor(idx_train)
                            idx_val = torch.tensor(idx_val)
                            idx_test = torch.tensor(idx_test)





                        train_label=labels[idx_train]
                        val_label = labels[idx_val]
                        test_label = labels[idx_test]
                        print(train_label)




                        with open('{}/{}/g_{}_l_{}_f_{}/{}_train_label.pkl'.format(path,data,graph_noise,label_noise,feature_noise,data), 'wb') as f:
                            pickle.dump(train_label, f)
                        #print(train_label.head(3))
                        with open('{}/{}/g_{}_l_{}_f_{}/{}_val_label.pkl'.format(path,data,graph_noise,label_noise,feature_noise,data), 'wb') as f:
                            pickle.dump(val_label, f)
                        #print(val_label.head(3))
                        with open('{}/{}/g_{}_l_{}_f_{}/{}_test_label.pkl'.format(path,data,graph_noise,label_noise,feature_noise,data), 'wb') as f:
                            pickle.dump(test_label, f)
                        with open('{}/{}/g_{}_l_{}_f_{}/{}_train_index.pkl'.format(path,data,graph_noise,label_noise,feature_noise,data), 'wb') as f:
                            pickle.dump(idx_train, f)
                        #print(train_label.head(3))
                        with open('{}/{}/g_{}_l_{}_f_{}/{}_val_index.pkl'.format(path,data,graph_noise,label_noise,feature_noise,data), 'wb') as f:
                            pickle.dump(idx_val, f)
                        #print(val_label.head(3))
                        with open('{}/{}/g_{}_l_{}_f_{}/{}_test_index.pkl'.format(path,data,graph_noise,label_noise,feature_noise,data), 'wb') as f:
                            pickle.dump(idx_test, f)
                        # with open('{}/g_{}_l_{}_f_{}/{}_adj.pkl'.format(data,graph_noise,label_noise,feature_noise,data), 'wb') as f:
                        #     pickle.dump(adj, f)
                        with open('{}/{}/g_{}_l_{}_f_{}/{}_all_label.pkl'.format(path,data,graph_noise,label_noise,feature_noise,data), 'wb') as f:
                            pickle.dump(labels, f)
                        with open('{}/{}/g_{}_l_{}_f_{}/{}_features.pkl'.format(path,data,graph_noise,label_noise,feature_noise,data), 'wb') as f:
                            pickle.dump(features, f)

                        #print(test_label.head(3))




                        #随机修改标签

                        n_perturbations=int(graph_noise*num_nodes)
                        print(n_perturbations)
                        train_label = np.array(train_label)
                        n_train_labels,_=Noi_ab.noisify_with_P(train_label,nb_classes,label_noise,noise_type=noise_type)
                        n_train_labels = pd.DataFrame(n_train_labels)
                        


                        with open('{}/{}/g_{}_l_{}_f_{}/{}_train_label_new_{}.pkl'.format(path,data,graph_noise,label_noise,feature_noise,data,label_noise), 'wb') as f:
                            pickle.dump(n_train_labels, f)
                        print(n_train_labels.head(3))





