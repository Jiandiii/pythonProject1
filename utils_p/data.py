import numpy as np
from numpy.testing import assert_array_almost_equal
import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, dense_to_sparse, to_dense_adj, is_undirected, remove_self_loops, degree, to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.transforms import NormalizeFeatures
from deeprobust.graph.data import Dataset, PrePtbDataset
import json
import scipy.sparse as sp
from copy import deepcopy
from deeprobust.graph.global_attack import Random
from deeprobust.graph import utils
from torch_geometric.datasets import Amazon, Coauthor, WikiCS
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

dataset_class = {'photo':Amazon, 'computers':Amazon, 'cs':Coauthor, 'physics':Coauthor}

def get_data(root, name, noise_type, noise_rate, label_rate=0.1):
    ## Load Base Data
    if name in ['cora', 'citeseer', 'pubmed', 'polblogs']:
        data = Dataset(root=root, name=name, setting='prognn')
        adj, features, labels = data.adj, data.features.toarray(), data.labels
    elif name in ['wikics', 'photo', 'computers', 'cs', 'physics']:
        if name == 'wikics':
            data = WikiCS(root)[0]
        else:
            data = dataset_class[name](root, name)[0]
        
        if name in ['wikics', 'photo', 'computers']:
            if os.path.isfile(os.path.join(root, f'{name}_lcc.pt')):
                data = torch.load(os.path.join(root, f'{name}_lcc.pt'))
                print('Load LCC')
            else:
                print('Extracting LCC')
                data = keep_only_largest_connected_component(data)
                torch.save(data, os.path.join(root, f'{name}_lcc.pt'))
                print('Save LCC')
            
        data.edge_index, _ = remove_self_loops(data.edge_index)
        data.edge_index = to_undirected(data.edge_index)
        adj = utils.to_scipy(to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0))[0])
        features = data.x.numpy()
        labels = data.y.long().numpy()
    
    elif 'auto' == name:
        if noise_type == 'clean':
            return torch.load(os.path.join(root, f'Automotive_clean_n_per_class5.pt'))
        elif 'fdn' in noise_type:
            return torch.load(os.path.join(root, f'Automotive_fdgn{noise_rate}_n_per_class5.pt'))
        else:
            return None
    elif 'garden' == name:
        if noise_type == 'clean':
            return torch.load(os.path.join(root, f'Patio_Lawn_and_Garden_clean_n_per_class5.pt'))
        elif 'fdn' in noise_type:
            return torch.load(os.path.join(root, f'Patio_Lawn_and_Garden_fdgn{noise_rate}_n_per_class5.pt'))
        else:
            return None
    
    ## Load Data Split
    if os.path.isfile(os.path.join(root, f'{name}_split_{label_rate}.pt')):
        splits = torch.load(os.path.join(root, f'{name}_split_{label_rate}.pt'))
    else:
        print('Creating Split')
        splits = create_split(labels, label_rate, 0.2-label_rate)
        torch.save(splits, os.path.join(root, f'{name}_split_{label_rate}.pt'))

    idx_train, idx_val, idx_test = splits['train'], splits['val'], splits['test']

    if noise_type == 'clean':
        dataset = Data()
        dataset.x = torch.from_numpy(features).float()
        dataset.y = torch.from_numpy(labels).long()
        dataset.edge_index = dense_to_sparse(torch.from_numpy(adj.toarray()))[0].long()
        dataset.edge_index = to_undirected(dataset.edge_index)
        
        dataset.train_mask = index_to_mask(torch.from_numpy(idx_train), dataset.x.size(0))
        dataset.val_mask = index_to_mask(torch.from_numpy(idx_val), dataset.x.size(0))
        dataset.test_mask = index_to_mask(torch.from_numpy(idx_test), dataset.x.size(0))
        torch.save(dataset, os.path.join(root, f'{name}_clean_graph.pt'))
        return dataset

    ## Load Noise Data
    elif 'fdn_fisn' in noise_type:
        if os.path.isfile(os.path.join(root, f'{name}_{noise_rate}_{noise_type}_{label_rate}_graph.pt')):
            print('Load FDGN FISN')
            data = torch.load(os.path.join(root, f'{name}_{noise_rate}_{noise_type}_{label_rate}_graph.pt'))
        else:
            print('Create FDGN FISN')
            data = generate_feat_dependent_noise(root, name, noise_rate, noise_type, features, adj, labels, idx_train, idx_val, idx_test)#随机修改feature
            fisn_rate = int(noise_type.split('fisn')[-1])
            num_rand_add_edge = int((data.edge_index.size(1) - adj.sum()) * fisn_rate / 100 ) // 2
            edge_index = data.edge_index.clone()
            data.edge_index, edge_index_to_add = add_random_edge(edge_index, num_rand_add_edge, num_nodes=data.x.size(0))#随机增加边
            torch.save(edge_index_to_add, os.path.join(root, f'{name}_{noise_rate}_{noise_type}_{label_rate}_random_structure.pt'))
            torch.save(data, os.path.join(root, f'{name}_{noise_rate}_{noise_type}_{label_rate}_graph.pt'))
        return data
    
    elif 'fin_all' in noise_type:    
        if os.path.isfile(os.path.join(root, f'{name}_{noise_rate}_{noise_type}_{label_rate}_graph.pt')):
            data = torch.load(os.path.join(root, f'{name}_{noise_rate}_{noise_type}_{label_rate}_graph.pt'))
        else:
            data = generate_feat_independent_noise(name, noise_rate, noise_type, features, adj, labels, idx_train, idx_val, idx_test)
            torch.save(data, os.path.join(root, f'{name}_{noise_rate}_{noise_type}_{label_rate}_graph.pt'))
        return data
    
    else:
        return None
    
def generate_feat_dependent_noise(root, name, noise_rate, noise_type, features, adj, labels, idx_train, idx_val, idx_test):
    # feature noise generation
    noise_rate /= 100
    features_, adj_, labels_, idx_train_, idx_val_, idx_test_ = deepcopy(features), deepcopy(adj), deepcopy(labels), deepcopy(idx_train), deepcopy(idx_val), deepcopy(idx_test)
    if os.path.isfile(os.path.join(root, f'{name}_feat_{noise_rate}_node.pt')):
        feat_attacked_node_train = torch.load(os.path.join(root, f'{name}_feat_{noise_rate}_node_train.pt'))
        feat_attacked_node_val=torch.load(os.path.join(root, f'{name}_feat_{noise_rate}_node_val.pt'))
        feat_attacked_node_test = torch.load(os.path.join(root, f'{name}_feat_{noise_rate}_node_test.pt'))
        feat_attacked_node=np.sort(np.concatenate((feat_attacked_node_train,feat_attacked_node_val,feat_attacked_node_test)))
        feat_attacked_mask = np.in1d(np.arange(len(labels_)), feat_attacked_node)
    else:
        noisytr= np.random.choice(idx_train_, int(len(idx_train_)*noise_rate), replace=False)
        noisyvl= np.random.choice(idx_val_, int(len(idx_val_)*noise_rate), replace=False)
        noisyts= np.random.choice(idx_test_, int(len(idx_test_)*noise_rate), replace=False)
        # feat_attacked_node = np.random.choice(np.arange(len(labels_)), int(len(labels_)*noise_rate), replace=False)
        # feat_attacked_node_train = np.sort(np.concatenate((noisytr, noisyvl, noisyts)))
        feat_attacked_node_train=np.sort(noisytr)
        feat_attacked_mask_train = np.in1d(np.arange(len(labels_)), feat_attacked_node_train)
        torch.save(feat_attacked_node_train, os.path.join(root, f'{name}_feat_{noise_rate}_feat_node_train.pt'))#记录注入噪声的node
        # feat_attacked_node_train = np.sort(np.concatenate((noisytr, noisyvl, noisyts)))
        feat_attacked_node_val = np.sort(noisyvl)
        feat_attacked_mask_val = np.in1d(np.arange(len(labels_)), feat_attacked_node_val)
        torch.save(feat_attacked_node_val,os.path.join(root, f'{name}_feat_{noise_rate}_feat_node_val.pt'))  # 记录注入噪声的node
        # feat_attacked_node_test = np.sort(np.concatenate((noisytr, noisyvl, noisyts)))
        feat_attacked_node_test = np.sort(noisyts)
        feat_attacked_mask_test = np.in1d(np.arange(len(labels_)), feat_attacked_node_test)
        torch.save(feat_attacked_node_test,os.path.join(root, f'{name}_feat_{noise_rate}_feat_node_test.pt'))  # 记录注入噪声的node
        feat_attacked_node = np.sort(
            np.concatenate((feat_attacked_node_train, feat_attacked_node_val, feat_attacked_node_test)))
        torch.save(feat_attacked_node_test, os.path.join(root, f'{name}_feat_{noise_rate}_feat_node.pt'))
        feat_attacked_mask = np.in1d(np.arange(len(labels_)), feat_attacked_node)
    if name in ['wikics', 'arxiv']:
        print('Creating Gaussian Noisy Features')
        perturb = torch.randn(features_.shape).numpy()
        features_[feat_attacked_mask] = features_[feat_attacked_mask] + perturb[feat_attacked_mask]*1
    elif name not in ['wikics', 'arxiv']:
        print('Creating Flipping Noisy Features')
        p_train = torch.from_numpy(features_[idx_train_].mean(1).reshape(-1, 1) * np.ones_like(features_[idx_train_]))
        p_val = torch.from_numpy(features_[idx_val_].mean(1).reshape(-1, 1) * np.ones_like(features_[idx_val_]))
        p_test = torch.from_numpy(np.random.uniform(0,1,size=features_[idx_test_].shape[0]).reshape(-1,1) * np.ones_like(features_[idx_test_]))
        perturb_train = p_train.bernoulli().numpy()
        perturb_val = p_val.bernoulli().numpy()
        perturb_test = p_test.bernoulli().numpy()
        features_[feat_attacked_mask_train] = perturb_train[feat_attacked_mask_train[idx_train_]]
        features_[feat_attacked_mask_val] = perturb_val[feat_attacked_mask_val[idx_val_]]
        features_[feat_attacked_mask_test] = perturb_test[feat_attacked_mask_test[idx_test_]]
        torch.save(features_, os.path.join(root, f'{name}_feat_{noise_rate}_feat.pt'))

    #################################################
    ########## Structure noise generation ###########
    #################################################

    print('Creating Feature Depedent Noisy Structure')
    deg = adj_.toarray().sum(0) 
    add_edge_lists = []
    for node in tqdm(feat_attacked_node):
        degree_node = round(deg[node].item() * 0.51)#生成几条边
        src = np.array([node]*degree_node)
        sim = cosine_similarity(features_[[node]], features_).squeeze()
        dst = np.argsort(sim)[-(degree_node+1):-1]#选出相似度最高的几组边
        add_edges = np.stack((src, dst))    
        add_edge_lists.append(add_edges)
    edge_noise = torch.from_numpy(np.concatenate(add_edge_lists, axis=1)).long()
    torch.save(edge_noise, os.path.join(root, f'{name}_feat_{noise_rate}_edge.pt'))
    tmp_adj_ = dense_to_sparse(torch.from_numpy(adj_.toarray()))[0].long()
    edge_index = to_undirected(torch.cat((tmp_adj_, edge_noise), dim=1))
    edge_index = remove_self_loops(edge_index)[0]

    #################################################
    ########## Label noise generation ###########
    #################################################    

    print('Creating Feature Depedent Noisy Labels')
    import pandas as pd
    df = pd.DataFrame({'src':edge_index[0],
                        'dst':labels_[edge_index[1]].squeeze()})
    df = pd.concat([df, pd.get_dummies(df.dst, prefix='class')], axis=1).drop(['dst'], axis=1)

    numer = df.groupby('src').sum()
    denom = df.groupby('src').size()
    transition_prob = (numer.values / denom.values[:, np.newaxis])

    clean_adj = torch.from_numpy(adj_.toarray())
    noisy_adj = to_dense_adj(edge_index, max_num_nodes=transition_prob.shape[0])[0]
    noisy_str_mask = clean_adj != noisy_adj
    noisy_str_node = noisy_str_mask.sum(1) > 0#找到添加结构噪声影响到的点
    union = torch.from_numpy(feat_attacked_mask) | noisy_str_node#联合两部分
    torch.save(noisy_str_node, os.path.join(root, f'{name}_feat_{noise_rate}_edge_node.pt'))
    torch.save(union, os.path.join(root, f'{name}_feat_{noise_rate}_union.pt'))
    node_list=[]

    for node in union.nonzero().flatten().numpy():
        if node in idx_train_ or node in idx_val_:
            labels_[node] = np.random.multinomial(1, transition_prob[node]).argmax()#只在训练集和验证集注入噪声
            node_list.append(node)
    torch.save(labels_, os.path.join(root, f'{name}_feat_{noise_rate}_label.pt'))
    torch.save(node_list, os.path.join(root, f'{name}_feat_{noise_rate}_label_node.pt'))

    dataset = Data()
    dataset.x = torch.from_numpy(features_).float()
    dataset.y = torch.from_numpy(labels_).long().squeeze()
    dataset.edge_index = edge_index
    
    dataset.train_mask = index_to_mask(torch.from_numpy(idx_train_), dataset.x.size(0))
    dataset.val_mask = index_to_mask(torch.from_numpy(idx_val_), dataset.x.size(0))
    dataset.test_mask = index_to_mask(torch.from_numpy(idx_test_), dataset.x.size(0))
    return dataset 

def generate_feat_independent_noise(name, noise_rate, noise_type, features, adj, labels, idx_train, idx_val, idx_test):
    # feature noise generation
    features_, adj_, labels_, idx_train_, idx_val_, idx_test_ = deepcopy(features), deepcopy(adj), deepcopy(labels), deepcopy(idx_train), deepcopy(idx_val), deepcopy(idx_test)
    noisytr= np.random.choice(idx_train, int(len(idx_train)*noise_rate/100), replace=False)
    noisyvl= np.random.choice(idx_val, int(len(idx_val)*noise_rate/100), replace=False)
    noisyts= np.random.choice(idx_test, int(len(idx_test)*noise_rate/100), replace=False)
    feat_attacked_node = np.sort(np.concatenate((noisytr, noisyvl, noisyts)))
    feat_attacked_mask = np.in1d(np.arange(len(labels)), feat_attacked_node)
    
    if name in ['wikics', 'arxiv']:
        print('Creating Gaussian Noisy Features')
        perturb = torch.randn(features_.shape).numpy()
        features_[feat_attacked_mask] = features_[feat_attacked_mask] + perturb[feat_attacked_mask]*1
    elif name not in ['wikics', 'arxiv']:
        print('Creating Flipping Noisy Features')
        p = torch.from_numpy(features_.mean(1).reshape(-1, 1) * np.ones_like(features_))
        perturb = p.bernoulli().numpy()
        features_[feat_attacked_mask] = perturb[feat_attacked_mask]
    
    attacker = Random()
    n_perturbations = int(noise_rate/100 * (adj.sum()//2))
    attacker.attack(adj_, n_perturbations, type='add')

    adj_ = attacker.modified_adj
    
    ptb_rate = noise_rate/100
    nclass = labels_.max() + 1
    train_labels = labels_[idx_train_]
    val_labels = labels_[idx_val_]
    train_val_labels = np.concatenate([train_labels,val_labels], axis=0)
    idx = np.concatenate([idx_train_, idx_val_],axis=0)
    noise_y, P, noise_idx, clean_idx = noisify_with_P(train_val_labels, idx_train_.shape[0], nclass, ptb_rate, 1995, 'unif')
    noise_idx, clean_idx = noise_idx, clean_idx
    labels_[idx] = noise_y

    dataset = Data()  
    dataset.x = torch.from_numpy(features_).float()
    dataset.edge_index, edge_attr = from_scipy_sparse_matrix(adj_)
    dataset.edge_index = to_undirected(dataset.edge_index)
    dataset.y = torch.from_numpy(labels_).long()
    dataset.train_mask = index_to_mask(torch.from_numpy(idx_train), dataset.x.size(0))
    dataset.val_mask = index_to_mask(torch.from_numpy(idx_val), dataset.x.size(0))
    dataset.test_mask = index_to_mask(torch.from_numpy(idx_test), dataset.x.size(0))

    return dataset

def generate_feat_noise(fdn, features, adj, labels, idx_train, idx_val, idx_test):
    # feature noise generation
    features_, adj_, labels_, idx_train_, idx_val_, idx_test_ = deepcopy(features), deepcopy(adj), deepcopy(labels), deepcopy(idx_train), deepcopy(idx_val), deepcopy(idx_test)
    data = fdn.clone()
    data.edge_index = dense_to_sparse(torch.from_numpy(adj_.toarray()))[0].long()
    data.edge_index = to_undirected(data.edge_index)
    data.y = torch.from_numpy(labels_).long()
    return data

def create_split(labels, tr, vl):
    tst = 1-tr-vl
    split = {}

    val_size = int(labels.shape[0] * vl)
    test_size = int(labels.shape[0] * tst)

    perm = np.random.permutation(labels.shape[0])
    test_index = perm[:test_size]
    val_index = perm[test_size:test_size + val_size]
    train_index = perm[test_size + val_size:]
    split['train'] = train_index
    split['val'] = val_index
    split['test'] = test_index

    return split

def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = np.float64(noise) / np.float64(size-1) * np.ones((size, size))
    np.fill_diagonal(P, (np.float64(1)-np.float64(noise))*np.ones(size))
    
    diag_idx = np.arange(size)
    P[diag_idx,diag_idx] = P[diag_idx,diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def build_pair_p(size, noise):
    assert(noise >= 0.) and (noise <= 1.)
    P = (1.0 - np.float64(noise)) * np.eye(size)
    for i in range(size):
        P[i,i-1] = np.float64(noise)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def noisify_with_P(y_train, train_num,  nb_classes, noise, random_state=None,  noise_type='uniform'):

    if noise > 0.0:
        if noise_type=='unif':
            print('Uniform noise')
            P = build_uniform_P(nb_classes, noise)
        elif noise_type == 'pair':
            print('Pair noise')
            P = build_pair_p(nb_classes, noise)
        else:
            print('Noise type have implemented')
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        y_train_noisy_l = y_train_noisy[:train_num]
        y_train_l = y_train[:train_num]
        noisy_idx = np.where(y_train_noisy_l-y_train_l!=0)[0]
        clean_idx = np.where(y_train_noisy_l-y_train_l==0)[0]
        assert actual_noise > 0.0
        y_train = y_train_noisy
    else:
        P = np.eye(nb_classes)
        noisy_idx = 0
        clean_idx = 0


    return y_train, P, noisy_idx, clean_idx


def index_to_mask(index, size = None):
    r"""Converts indices to a mask representation.

    Args:
        idx (Tensor): The indices.
        size (int, optional). The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

def keep_only_largest_connected_component(data):
    lcc = get_largest_connected_component(data)

    x_new = data.x[lcc]
    y_new = data.y[lcc]

    row, col = data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
        x=x_new,
        edge_index=torch.LongTensor(edges),
        y=y_new,
    )

    return data


def get_component(data, start=0):
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(data):
    remaining_nodes = set(range(data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(data, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch_geometric.utils import negative_sampling

def add_random_edge(
    edge_index,
    num_add: int = 0,
    force_undirected: bool = False,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tuple[Tensor, Tensor]:

    # if p < 0. or p > 1.:
    #     raise ValueError(f"Ratio of added edges has to be between 0 and 1 "
    #                      f"(got '{p}')")
    # if force_undirected and isinstance(num_nodes, (tuple, list)):
    #     raise RuntimeError("'force_undirected' is not supported for "
    #                        "bipartite graphs")

    device = edge_index.device
    if num_add == 0:
        edge_index_to_add = torch.tensor([[], []], device=device)
        return edge_index, edge_index_to_add

    edge_index_to_add = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_add,
        force_undirected=force_undirected,
    )

    edge_index = torch.cat([edge_index, edge_index_to_add], dim=1)


    return to_undirected(edge_index), edge_index_to_add
