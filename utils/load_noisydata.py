import pickle
import scipy
import os

def load_noisydata(noise_type, dataset, graph_noise, label_noise, feature_noise, part_id=None):
    if noise_type == 'uniform':
        datapath = './new/{}/g_{}_l_{}_f_{}/'.format(dataset, graph_noise, label_noise, feature_noise)
    elif noise_type == 'pair_noise':
        datapath = './new_pair/{}/g_{}_l_{}_f_{}/'.format(dataset, graph_noise, label_noise, feature_noise)
    elif noise_type == 'class_confusion':
        datapath = './class_confusion/{}/g_{}_l_{}_f_{}/'.format(dataset, graph_noise, label_noise, feature_noise)
    elif noise_type == 'feature_based':
        datapath = './feature_based/{}/g_{}_l_{}_f_{}/'.format(dataset, graph_noise, label_noise, feature_noise)
    else:
        datapath = './new/{}/g_{}_l_{}_f_{}/'.format(dataset, graph_noise, label_noise, feature_noise)

    part_suffix = f'_part{part_id}' if part_id is not None else ''

    with open(datapath + '{}_train_index{}.pkl'.format(dataset, part_suffix), 'rb') as f:
        train_idx = pickle.load(f, encoding='latin1')
    with open(datapath + '{}_val_index{}.pkl'.format(dataset, part_suffix), 'rb') as f:
        val_idx = pickle.load(f, encoding='latin1')
    with open(datapath + '{}_test_index{}.pkl'.format(dataset, part_suffix), 'rb') as f:
        test_idx = pickle.load(f, encoding='latin1')
    with open(datapath + '{}_train_label{}.pkl'.format(dataset, part_suffix), 'rb') as f:
        true_train_label = pickle.load(f, encoding='latin1')
    with open(datapath + '{}_train_label_new_{}{}.pkl'.format(dataset, label_noise, part_suffix), 'rb') as f:
        train_label = pickle.load(f, encoding='latin1')
    with open(datapath + '{}_val_label{}.pkl'.format(dataset, part_suffix), 'rb') as f:
        val_label = pickle.load(f, encoding='latin1')
    with open(datapath + '{}_test_label{}.pkl'.format(dataset, part_suffix), 'rb') as f:
        test_label = pickle.load(f, encoding='latin1')
    with open(datapath + '{}_features{}.pkl'.format(dataset, part_suffix), 'rb') as f:
        features = pickle.load(f, encoding='latin1')
    
    adj_path = datapath + '{}_mod_adj_add_{}{}.npz'.format(dataset, graph_noise, part_suffix)
    adj = scipy.sparse.load_npz(adj_path)
    
    with open(datapath + 'nclass{}.pkl'.format(part_suffix), 'rb') as f:
        num_classes = pickle.load(f, encoding='latin1')
    with open(datapath + '{}_all_label{}.pkl'.format(dataset, part_suffix), 'rb') as f:
        labels = pickle.load(f, encoding='latin1')

    return train_idx, val_idx, test_idx, train_label, val_label, test_label, features, adj, true_train_label, num_classes, labels

