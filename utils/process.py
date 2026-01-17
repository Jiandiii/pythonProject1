import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, diags
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset,Evaluator
from torch_geometric.utils import to_undirected
import os.path as osp
from torch_geometric.datasets import Planetoid, Amazon
import dgl
import os
import json
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from ogb.graphproppred import DglGraphPropPredDataset
from dgl.dataloading import GraphDataLoader

import random

def get_node_graph_mapping(graph):
    # 获取每个图的节点数
    batch_num_nodes = graph.batch_num_nodes()
    
    # 创建映射数组
    mapping = []
    for graph_idx, num_nodes in enumerate(batch_num_nodes):
        # 为每个节点标记它属于哪个图
        mapping.extend([graph_idx] * num_nodes)
    
    # mapping = torch.tensor(mapping)
    # print(f"节点 0-{len(mapping)-1}的图索引: {mapping}")
    
    return mapping

def edge_index_to_adj(edge_index, num_nodes=None):
    # 如果没有指定节点数，从 edge_index 推断
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    
    # 创建空的邻接矩阵
    adj = torch.zeros((num_nodes, num_nodes))
    
    # 根据 edge_index 填充邻接矩阵
    adj[edge_index[0], edge_index[1]] = 1
    
    return adj


def _collate_fn(batch):
    # 小批次是一个元组(graph, label)列表
    graphs = [e[0] for e in batch]
    g = dgl.batch(graphs)
    labels = [e[1] for e in batch]
    labels = torch.stack(labels, 0)
    return g, labels


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
def load_single_graph(gra_noi_rat ,l,dataset_N,type):
    if dataset_N in ['CiteSeer', 'PubMed', 'Photo', 'Computers','ogbn-arxiv','facebook','ogbn-products','ogbn-proteins']:
        if dataset_N in ['CiteSeer', 'PubMed']:
            # path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/')
            path = osp.join('./', 'tmp/')
            dataset = Planetoid(path,dataset_N )
        elif dataset_N in ['facebook']:
            dataset = torch.load(osp.join('./', 'facebook_large/', 'musae_facebook_data.pt'))
        elif dataset_N in ['Photo', 'Computers']:
            # path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/')
            path = osp.join('./', 'tmp/')
            dataset = Amazon(path, dataset_N, pre_transform=None)  # transform=T.ToSparseTensor(),
        elif dataset_N in ['ogbn-arxiv']:
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/')
        elif dataset_N in ['ogbn-products']:
            dataset = PygNodePropPredDataset(name='ogbn-products', root='./products/')
        elif dataset_N in ['ogbn-proteins']:
            dataset = PygNodePropPredDataset(name='ogbn-proteins', root='./proteins/')
            # PygNodePropPredDataset()
            # print(dataset[0])
            # print(dataset[0].adj_t)
            # row, col, _ = dataset_old[0].adj_t.coo() # 获取COO格式的行列索引
            # edge_index = torch.stack([row, col], dim=0)
            # dataset_new = Data(
            #     x=dataset_old[0].x,
            #     edge_index=edge_index,  # 新增的 edge_index
            #     y=dataset_old[0].y,
            #     node_year=dataset_old[0].node_year,
            #     adj_t=dataset_old[0].adj_t,    # 保留原始稀疏矩阵
            #     num_nodes=dataset_old[0].num_nodes
            # )

        # data = dataset[0]
        if dataset_N in ['facebook']:
            data = dataset
        else:
            data = dataset[0]

        # ogbn-proteins does not have node features by default
        if data.x is None:
            if dataset_N == 'ogbn-proteins':
                # Use node degrees or constant features as dummy node features
                from torch_geometric.utils import degree
                deg = degree(data.edge_index[0], data.num_nodes).view(-1, 1)
                data.x = deg
            else:
                data.x = torch.ones((data.num_nodes, 1))

        if 'train_mask' not in data:
            num_nodes = data.num_nodes if data.x is None else data.x.size(0)
            idx_train = range(int(0.05*num_nodes))
            idx_val = range(int(0.05*num_nodes), int(0.05*num_nodes)+int(0.1*num_nodes))
            idx_test = range( int(0.65*num_nodes), num_nodes)
            idx_train = torch.LongTensor(idx_train)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)
        else:
            idx_train = data.train_mask
            idx_val = data.val_mask
            idx_test = data.test_mask

        i = torch.Tensor.long(data.edge_index)
        v = torch.FloatTensor(torch.ones([data.num_edges]))
        A_sp = torch.sparse.FloatTensor(i, v, torch.Size([data.num_nodes, data.num_nodes]))

        if gra_noi_rat != 0:
            # Directly use sparse matrix operations without converting to dense
            A = SparseTensor(row=i[0], col=i[1], value=v, sparse_sizes=(data.num_nodes, data.num_nodes))
            import utils.random as UR
            attacker = UR.Random()
            if dataset_N in ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins']:
                n_perturbations = int(gra_noi_rat * A.nnz())
            else:
                n_perturbations = int(gra_noi_rat * (A.nnz() // 2))
            perturbed_adj = attacker.attack(A.to_scipy(), n_perturbations, data.x, gra_noi_rat, l, dataset, type=type)
            adj = sp.coo_matrix(perturbed_adj)
            values = adj.data
            indices = np.vstack((adj.row, adj.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            A_sp = torch.sparse.FloatTensor(i, v, torch.Size(adj.shape))

        if dataset_N in ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins']:
            # 假设 adj_t 是 SparseTensor
            # A = sp.coo_matrix(A)
            indices = A_sp.coalesce().indices()  # 形状 [2, nnz]，包含 (row, col)
            value = A_sp.coalesce().values()    # 形状 [nnz]，包含边的权重（这里全是1）

            row, col = indices[0], indices[1]  # 分离行和列索引

            # 计算每行的度数（degree）
            degree = A_sp.sum(dim=1).to_dense().view(-1)  # [num_nodes]

            # 归一化：A_{ij} / degree[i]
            normalized_value = value / degree[row]  # 除以每行的度数

            # 构建归一化的稀疏矩阵
            A_nomal = SparseTensor(
                row=row, 
                col=col, 
                value=normalized_value,
                sparse_sizes=A_sp.size()
            )
            
            N = A_sp.size(0)  # 节点数量

            # 构造自环的 indices 和 values
            self_loop_indices = torch.stack([
                torch.arange(N, device=A_sp.device),
                torch.arange(N, device=A_sp.device),
            ], dim=0)
            self_loop_values = torch.ones(N, device=A_sp.device)

            # 合并原始矩阵和自环
            new_indices = torch.cat([indices, self_loop_indices], dim=1)
            new_values = torch.cat([value, self_loop_values])

            # 构建新稀疏矩阵
            A_I = torch.sparse_coo_tensor(
                indices=new_indices,
                values=new_values,
                size=A_sp.size(),
            )
            indices = A_I.coalesce().indices()  # 形状 [2, nnz]，包含 (row, col)
            value = A_I.coalesce().values()    # 形状 [nnz]，包含边的权重（这里全是1）
            row, col = indices[0], indices[1]  # 分离行和列索引

            # 计算每行的度数（degree）
            degree = A_I.sum(dim=1).to_dense().view(-1)  # [num_nodes]

            # 归一化：A_{ij} / degree[i]
            normalized_value = value / degree[row]  # 除以每行的度数

            # 构建归一化的稀疏矩阵
            A_I_nomal = SparseTensor(
                row=row, 
                col=col, 
                value=normalized_value,
                sparse_sizes=A_I.size()
            )
            label = data.y
        else:
            A_nomal = row_normalize_sparse(A_sp)
            I = SparseTensor.eye(A_sp.size(0))
            A_I = A_sp + I
            A_I_nomal = row_normalize_sparse(A_I)
            label = data.y

        return [A_I_nomal, A_nomal, A_nomal], data.x, label, idx_train, idx_val, idx_test

    elif dataset_N in ['Cora']:
        idx_features_labels = np.genfromtxt("{}{}.content".format("./tmp/Cora/", "cora"), dtype=np.dtype(str))

        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        def normalize_features(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format("./tmp/Cora/", "cora"), dtype=np.int32)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
            edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize_features(features)
        features = torch.FloatTensor(np.array(features.todense()))
        
        if gra_noi_rat !=0:
            import utils.random as UR
            attacker = UR.Random()
            A = adj
            n_perturbations = int(gra_noi_rat * (A.sum() // 2))
            perturbed_adj = attacker.attack(A, n_perturbations,features,gra_noi_rat,l,dataset_N, type=type)
            adj = sp.coo_matrix(perturbed_adj)

        A_I_nomal = normalize_adj(adj + sp.eye(adj.shape[0]))
        A_nomal = normalize_adj(adj)
        A =  adj.todense()

        splits=create_split(labels,0.1,0.1)
        idx_train, idx_val, idx_test = splits['train'], splits['val'], splits['test']


        # idx_train = range(140)
        # idx_val = range(200, 500)
        # idx_test = range(500, 1500)

        A_I_nomal = torch.FloatTensor(np.array(A_I_nomal.todense()))
        A_nomal = torch.FloatTensor(np.array(A_nomal.todense()))
        A = torch.FloatTensor(np.array(A))

        
        labels = torch.LongTensor(np.where(labels)[1])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return [A_I_nomal, A_nomal, A], features, labels, idx_train, idx_val, idx_test

def load_batch_graph(gra_noi_rat ,l,dataset_N,type):
    if dataset_N in ['molhiv']:
        # 载入数据集
        dataset = DglGraphPropPredDataset(name='ogbg-molhiv', root='./molhiv/')
        split_idx = dataset.get_idx_split()
        label_list=[]
        feature_list=[]
        A_I_nomal=[]
        node_to_graph_list=[]
        # dataloader
        train_loader = GraphDataLoader(dataset[split_idx["train"]], batch_size=64, shuffle=True, collate_fn=_collate_fn)
        valid_loader = GraphDataLoader(dataset[split_idx["valid"]], batch_size=64, shuffle=False, collate_fn=_collate_fn)
        test_loader = GraphDataLoader(dataset[split_idx["test"]], batch_size=64, shuffle=False, collate_fn=_collate_fn)
        for loader in [train_loader, valid_loader, test_loader]:
            for batch in loader:  # 遍历 DataLoader
                # batch 现在是一个批次的图数据
                batch_graphs = batch
                graph,label= batch_graphs
                
                node_to_graph = get_node_graph_mapping(graph)
                node_to_graph_list.append(node_to_graph)
                edge_index = graph.edges()
                num_edges = edge_index[0].shape[0]
                num_nodes = graph.num_nodes()
                node_features = graph.ndata['feat']
                edge_index = torch.stack(edge_index)
                i = torch.Tensor.long(edge_index)
                v = torch.FloatTensor(torch.ones([num_edges]))
                A_sp = torch.sparse.FloatTensor(i, v, torch.Size([num_nodes, num_nodes]))
                A = A_sp.to_dense()
                indices = torch.nonzero(A).t()  # 获取非零元素的索引 [2, nnz]
                values = A[A != 0]            # 获取非零值 [nnz]

                # 构建稀疏矩阵
                A_sp = torch.sparse_coo_tensor(
                    indices=indices,
                    values=values,
                    size=A.shape,
                )

                A = sp.csr_matrix(A)
                import utils.random as UR
                attacker = UR.Random()
                n_perturbations = int(gra_noi_rat * (A.sum() // 2))
                perturbed_adj = attacker.attack(A, n_perturbations,node_features,gra_noi_rat,l,dataset, type=type)
                adj = sp.coo_matrix(perturbed_adj)
                values = adj.data
                indices = np.vstack((adj.row, adj.col))
                i = torch.LongTensor(indices)
                v = torch.FloatTensor(values)
                shape = adj.shape
                A_sp = torch.sparse.FloatTensor(i, v, torch.Size(shape))
                A =A_sp.to_dense()
                indices = torch.nonzero(A).t()  # 获取非零元素的索引 [2, nnz]
                values = A[A != 0]            # 获取非零值 [nnz]
                # 构建稀疏矩阵
                A_sp = torch.sparse_coo_tensor(
                    indices=indices,
                    values=values,
                    size=A.shape,
                )
                
                label_list.extend(label)
                
                feature_list.extend(node_features)
                A_nomal = row_normalize_sparse(A)
                I = torch.eye(A.shape[1]).to(A.device)
                A_I = A + I
                A_I_nomal .append(row_normalize_sparse(A_I)) 

        label_list = torch.stack(label_list)
        label_list=torch.flatten(label_list)
        feature_list = torch.stack(feature_list)
        return A_I_nomal, feature_list, label_list,split_idx["train"],split_idx["valid"],split_idx["test"],node_to_graph_list,len(train_loader),len(valid_loader),len(test_loader),
        

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features,eps =1e-6):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = rowsum + eps
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    try:
        return features.todense()
    except:
        return features


def row_normalize(A):
    """Row-normalize dense matrix"""
    eps = 2.2204e-16
    rowsum = A.sum(dim=-1).clamp(min=0.) + eps
    r_inv = rowsum.pow(-1)
    A = r_inv.unsqueeze(-1) * A
    return A


def row_normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch2dgl(graph):
    N = graph.shape[0]
    if graph.is_sparse:
        graph_sp = graph.coalesce()
    else:
        graph_sp = graph.to_sparse()
    edges_src = graph_sp.indices()[0]
    edges_dst = graph_sp.indices()[1]
    edges_features = graph_sp.values()
    graph_dgl = dgl.graph((edges_src, edges_dst), num_nodes=N)
    # graph_dgl.edate['w'] = edges_features
    return graph_dgl

