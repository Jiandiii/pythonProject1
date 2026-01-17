import pickle
import scipy
import torch
def load_noisydata_gc(noise_type,dataset,graph_noise,label_noise,feature_noise):
    if noise_type=='uniform_gc':
        datapath='./new_gc/{}/g_{}_l_{}_f_{}/'.format(dataset,graph_noise,label_noise,feature_noise)
        # datapath = './pro/{}/g_{}_l_{}/'.format(dataset, graph_noise, label_noise)
    if noise_type=='pair_noise_gc':
        datapath='./new_pair_gc/{}/g_{}_l_{}_f_{}/'.format(dataset,graph_noise,label_noise,feature_noise)
    if noise_type=='feature_based_gc':
        datapath='./feature_based_gc/{}/g_{}_l_{}_f_{}/'.format(dataset,graph_noise,label_noise,feature_noise)
    with open(datapath+'{}_train_index.pkl'.format(dataset), 'rb') as f:
        train_idx = pickle.load(f, encoding='latin1')
    with open(datapath+'{}_val_index.pkl'.format(dataset), 'rb') as f:
        val_idx = pickle.load(f, encoding='latin1')
    with open(datapath+'{}_test_index.pkl'.format(dataset), 'rb') as f:
        test_idx = pickle.load(f, encoding='latin1')
    with open(datapath+'{}_train_label.pkl'.format(dataset,label_noise), 'rb') as f:
        true_train_label = pickle.load(f, encoding='latin1')
    with open(datapath+'{}_train_label_new_{}.pkl'.format(dataset,label_noise), 'rb') as f:
        train_label = pickle.load(f, encoding='latin1')
    with open(datapath+'{}_val_label.pkl'.format(dataset), 'rb') as f:
        val_label = pickle.load(f, encoding='latin1')
    with open(datapath+'{}_test_label.pkl'.format(dataset), 'rb') as f:
        test_label = pickle.load(f, encoding='latin1')
    with open(datapath+'{}_features.pkl'.format(dataset), 'rb') as f:
        features = pickle.load(f, encoding='latin1')
    with open(datapath+'{}_mod_adj_add_{}.pkl'.format(dataset,graph_noise), 'rb') as f:
        adj = pickle.load(f, encoding='latin1')
    # adj = scipy.sparse.load_npz(datapath+'{}_mod_adj_add_{}.npz'.format(dataset,graph_noise))
    with open(datapath+'nclass.pkl', 'rb') as f:
        num_classes = pickle.load(f, encoding='latin1')
    with open(datapath+'{}_all_label.pkl'.format(dataset), 'rb') as f:
        labels = pickle.load(f, encoding='latin1')
    with open(datapath+'node_to_graph.pkl'.format(dataset), 'rb') as f:
        node_to_graph = pickle.load(f, encoding='latin1')
    with open(datapath+'len_train.pkl'.format(dataset), 'rb') as f:
        len_train = pickle.load(f, encoding='latin1')
    with open(datapath+'len_val.pkl'.format(dataset), 'rb') as f:
        len_val = pickle.load(f, encoding='latin1')
    with open(datapath+'len_test.pkl'.format(dataset), 'rb') as f:
        len_test = pickle.load(f, encoding='latin1')
    noise_label_list=[]
    feature_list=[]
    noise_labels=labels.clone()
    new=torch.tensor(train_label.values.flatten())
    noise_labels[train_idx]=new
    for i in range(len(node_to_graph)):
        num_labels=max(node_to_graph[i])+1
        num_features=len(node_to_graph[i])
        noise_label_list.append(noise_labels[:num_labels])
        feature_list.append(features[:num_features])
        noise_labels=noise_labels[num_labels:]
        features=features[num_features:]
    train_batch=list(range(len_train))
    val_batch=list(range(len_train,len_train+len_val))
    test_batch=list(range(len_train+len_val,len_train+len_val+len_test))

    return train_batch,val_batch,test_batch,feature_list,adj,num_classes,noise_label_list

