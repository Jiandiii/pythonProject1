import os
import numpy as np
import pandas as pd
import torch
from utils import Noise_about as Noi_ab
import scipy.sparse as sp
import pickle
from utils import process
from utils import utils
from torch_sparse import SparseTensor

try:
    from torch_geometric.utils import metis as pyg_metis
except (ImportError, AttributeError):
    pyg_metis = None

try:
    import pymetis
except ImportError:
    pymetis = None

SUBGRAPH_NODE_COUNT = None
METIS_PARTITION_COUNT = 4


def _to_long_tensor(idx, total_nodes):
    if isinstance(idx, torch.Tensor):
        if idx.dtype == torch.bool:
            return torch.nonzero(idx, as_tuple=False).view(-1)
        return idx.long()
    if isinstance(idx, np.ndarray):
        return torch.from_numpy(idx.astype(np.int64))
    if isinstance(idx, range):
        return torch.tensor(list(idx), dtype=torch.long)
    if isinstance(idx, (list, tuple)):
        return torch.tensor(idx, dtype=torch.long)
    raise TypeError(f"Unsupported index type: {type(idx)}")


def _remap_indices(idx, mapping):
    idx = _to_long_tensor(idx, mapping.size(0))
    mapped = mapping[idx]
    return mapped[mapped >= 0]


def _adj_to_edge_index(adj, num_nodes):
    if isinstance(adj, SparseTensor):
        row, col, _ = adj.coo()
        return torch.stack((row, col), dim=0)
    if isinstance(adj, torch.Tensor):
        return adj.nonzero(as_tuple=False).t().contiguous()
    if sp.issparse(adj):
        coo = adj.tocoo()
        indices = np.vstack((coo.row, coo.col))
        return torch.from_numpy(indices).long()
    raise TypeError(f"Unsupported adjacency type for edge index: {type(adj)}")


def _subgraph_sparse_tensor(adj, nodes, mapping):
    adj = adj.coalesce()
    row, col, value = adj.coo()
    if mapping.device != row.device:
        mapping = mapping.to(row.device)
    node_mask = torch.zeros(mapping.size(0), dtype=torch.bool, device=row.device)
    node_mask[nodes.to(row.device)] = True
    edge_mask = node_mask[row] & node_mask[col]
    row = mapping[row[edge_mask]]
    col = mapping[col[edge_mask]]
    value = value[edge_mask]
    return SparseTensor(row=row, col=col, value=value, sparse_sizes=(nodes.numel(), nodes.numel()))


def _subgraph_matrix(adj, nodes):
    if isinstance(adj, torch.Tensor):
        return adj.index_select(0, nodes).index_select(1, nodes)
    if sp.issparse(adj):
        node_idx = nodes.cpu().numpy()
        return adj.tocsr()[node_idx][:, node_idx]
    if isinstance(adj, SparseTensor):
        mapping = torch.full((adj.size(0),), -1, dtype=torch.long, device=nodes.device)
        mapping[nodes] = torch.arange(nodes.size(0), device=nodes.device)
        return _subgraph_sparse_tensor(adj, nodes, mapping)
    raise TypeError(f"Unsupported adjacency type: {type(adj)}")


def _convert_adj_to_scipy(adj):
    if isinstance(adj, torch.Tensor):
        return utils.to_scipy(adj.cpu())
    if isinstance(adj, SparseTensor):
        row, col, value = adj.coo()
        return sp.coo_matrix((value.cpu().numpy(), (row.cpu().numpy(), col.cpu().numpy())), shape=adj.sparse_sizes())
    if sp.issparse(adj):
        return adj
    if isinstance(adj, np.ndarray):
        return sp.csr_matrix(adj)
    raise TypeError(f"Unsupported adjacency type for saving: {type(adj)}")


def extract_subgraph_by_nodes(adj_list, features, labels, idx_train, idx_val, idx_test, nodes):
    total_nodes = features.size(0)
    nodes = _to_long_tensor(nodes, total_nodes)
    if nodes.numel() == 0:
        return adj_list, features, labels, idx_train, idx_val, idx_test
    nodes = nodes.unique(sorted=True)
    features_sub = features[nodes]
    labels_sub = labels[nodes]
    mapping = torch.full((total_nodes,), -1, dtype=torch.long)
    mapping[nodes] = torch.arange(nodes.size(0), dtype=torch.long)
    adj_sub = [_subgraph_matrix(item, nodes) for item in adj_list]
    if isinstance(adj_list, tuple):
        adj_sub = tuple(adj_sub)
    idx_train_sub = _remap_indices(idx_train, mapping)
    idx_val_sub = _remap_indices(idx_val, mapping)
    idx_test_sub = _remap_indices(idx_test, mapping)
    return adj_sub, features_sub, labels_sub, idx_train_sub, idx_val_sub, idx_test_sub


def extract_subgraph(adj_list, features, labels, idx_train, idx_val, idx_test, num_nodes):
    if num_nodes is None:
        return adj_list, features, labels, idx_train, idx_val, idx_test
    total_nodes = features.size(0)
    if num_nodes >= total_nodes:
        return adj_list, features, labels, idx_train, idx_val, idx_test
    nodes = torch.randperm(total_nodes)[:num_nodes]
    return extract_subgraph_by_nodes(adj_list, features, labels, idx_train, idx_val, idx_test, nodes)


def partition_with_metis(adj_list, features, labels, idx_train, idx_val, idx_test, num_parts):
    base_result = [(0, adj_list, features, labels, idx_train, idx_val, idx_test)]
    if num_parts is None or num_parts <= 1:
        return base_result

    base_adj = adj_list[0]
    num_nodes = features.size(0)
    edge_index = _adj_to_edge_index(base_adj, num_nodes)
    if edge_index.numel() == 0:
        return base_result

    cluster = None
    # 优先使用 PyG 自带的 METIS
    if pyg_metis is not None:
        try:
            cluster, _ = pyg_metis(edge_index, num_nodes, num_parts)
        except Exception as e:
            print(f"PyG METIS 运行失败: {e}")

    # 如果 PyG 的不可用，尝试使用独立的 pymetis 库
    if cluster is None and pymetis is not None:
        print("正在使用 pymetis 进行图分块...")
        try:
            # 将 edge_index 转换为 pymetis 要求的邻接表格式
            adj_scipy = _convert_adj_to_scipy(base_adj).tocsr()
            adj_list_pymetis = [adj_scipy.indices[adj_scipy.indptr[i]:adj_scipy.indptr[i+1]].tolist() for i in range(num_nodes)]
            n_cuts, membership = pymetis.part_graph(num_parts, adjacency=adj_list_pymetis)
            cluster = torch.tensor(membership)
        except Exception as e:
            print(f"pymetis 运行失败: {e}")

    if cluster is None:
        print("警告: 未能找到可用的 METIS 实现。将跳过分块。请尝试 pip install pymetis")
        return base_result

    partitions = []
    for part_idx in range(num_parts):
        nodes = torch.nonzero(cluster == part_idx, as_tuple=False).view(-1)
        if nodes.numel() == 0:
            continue
        part = extract_subgraph_by_nodes(adj_list, features, labels, idx_train, idx_val, idx_test, nodes)
        partitions.append((part_idx, *part))
    return partitions or base_result


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l, dtype=bool)
    mask[idx] = 1
    return mask


if __name__ == '__main__':
    np.random.seed(2024)
    torch.manual_seed(2024)

    torch.cuda.manual_seed_all(2024)

    data_list = ['ogbn-arxiv']  # 'cora','citeseer',arxiv,proteins
    path_list = ['new']  # 'new_pair','class_confusion','feature_based','new'

    for data in data_list:
        label_noise_list = [0.0]
        graph_noise_list = [0.0]
        feature_noise_list = [0.0]

        for label_noise in label_noise_list:
            for graph_noise in graph_noise_list:
                for feature_noise in feature_noise_list:
                    for path in path_list:
                        noise_type = 'uniform' if path != 'new_pair' else 'pair'
                        print(data, path, label_noise, graph_noise, feature_noise, noise_type)

                        root_dir = os.path.join(path, data, f'g_{graph_noise}_l_{label_noise}_f_{feature_noise}')
                        os.makedirs(root_dir, exist_ok=True)

                        if path != 'feature_based':
                            adj, features, labels, idx_train, idx_val, idx_test = process.load_single_graph(
                                graph_noise, label_noise, data, type='add'
                            )
                        else:
                            adj, features, labels, idx_train, idx_val, idx_test = process.load_single_graph(
                                graph_noise, label_noise, data, type='feature_based'
                            )

                        desired_nodes = SUBGRAPH_NODE_COUNT
                        if isinstance(desired_nodes, dict):
                            desired_nodes = desired_nodes.get(data)
                        adj, features, labels, idx_train, idx_val, idx_test = extract_subgraph(
                            adj,
                            features,
                            labels,
                            idx_train,
                            idx_val,
                            idx_test,
                            desired_nodes,
                        )

                        desired_parts = METIS_PARTITION_COUNT
                        if isinstance(desired_parts, dict):
                            desired_parts = desired_parts.get(data)
                        partitions = partition_with_metis(
                            adj,
                            features,
                            labels,
                            idx_train,
                            idx_val,
                            idx_test,
                            desired_parts,
                        )

                        multiple_parts = len(partitions) > 1

                        for part_id, adj_part, feat_part, label_part, idx_train_part, idx_val_part, idx_test_part in partitions:
                            part_suffix = f'_part{part_id}' if multiple_parts else ''

                            features = feat_part.clone().cpu() if isinstance(feat_part, torch.Tensor) else torch.tensor(feat_part).clone()
                            labels = label_part.clone().cpu() if isinstance(label_part, torch.Tensor) else torch.tensor(label_part)
                            idx_train = _to_long_tensor(idx_train_part, features.size(0)).cpu()
                            idx_val = _to_long_tensor(idx_val_part, features.size(0)).cpu()
                            idx_test = _to_long_tensor(idx_test_part, features.size(0)).cpu()

                            if feature_noise != 0 and features.size(0) > 0:
                                noisy_count = int(features.size(0) * feature_noise)
                                if noisy_count > 0:
                                    noisy_indices = np.random.choice(features.size(0), noisy_count, replace=False)
                                    noisy_indices.sort()
                                    mask = torch.zeros(features.size(0), dtype=torch.bool)
                                    mask[torch.from_numpy(noisy_indices).long()] = True

                                    if path != 'class_confusion':
                                        feat_np = features.detach().cpu().numpy()
                                        mean_vec = feat_np.mean(1, keepdims=True)
                                        p = torch.from_numpy(mean_vec * np.ones_like(feat_np))
                                        if (p > 1).any() or (p < 0).any():
                                            perturb_np = np.random.uniform(feat_np.min(), feat_np.max(), size=features.shape)
                                        else:
                                            perturb_np = p.bernoulli().numpy()
                                        perturb = torch.from_numpy(perturb_np).to(features.dtype)
                                        features[mask] = perturb[mask]
                                    else:
                                        unique_classes = torch.unique(labels)
                                        for cls in unique_classes:
                                            cls_indices = (labels == cls).nonzero(as_tuple=False).view(-1)
                                            if cls_indices.numel() == 0:
                                                continue
                                            sample_size = int(cls_indices.numel() * feature_noise)
                                            if sample_size == 0:
                                                continue
                                            cls_indices_np = cls_indices.cpu().numpy()
                                            selected_np = np.random.choice(cls_indices_np, size=sample_size, replace=False)
                                            other_mask = labels != cls
                                            other_features = features[other_mask]
                                            if other_features.size(0) == 0:
                                                continue
                                            replace_np = np.random.choice(other_features.size(0), size=sample_size, replace=True)
                                            selected_tensor = torch.from_numpy(selected_np).long()
                                            replace_tensor = torch.from_numpy(replace_np).long()
                                            features[selected_tensor] = other_features[replace_tensor]

                            adj_current = adj_part[0]
                            scipy_adj = _convert_adj_to_scipy(adj_current)
                            adj_filename = os.path.join(root_dir, f'{data}_mod_adj_add_{graph_noise}{part_suffix}')
                            sp.save_npz(adj_filename, scipy_adj)

                            print(f'Partition {part_id}:', idx_train, idx_val, idx_test)

                            if data in ['ogbn-arxiv', 'ogbn-products'] and isinstance(labels, torch.Tensor):
                                labels = labels.flatten()

                            if isinstance(labels, torch.Tensor):
                                max_label = labels.max().item()
                                min_label = labels.min().item()
                            else:
                                max_label = float(np.max(labels))
                                min_label = float(np.min(labels))
                            nb_classes = int(max_label - min_label) + 1

                            with open(os.path.join(root_dir, f'nclass{part_suffix}.pkl'), 'wb') as f:
                                pickle.dump(nb_classes, f)

                            dim = features.shape[1]
                            with open(os.path.join(root_dir, f'dim{part_suffix}.pkl'), 'wb') as f:
                                pickle.dump(dim, f)

                            train_label = labels[idx_train]
                            val_label = labels[idx_val]
                            test_label = labels[idx_test]
                            print(train_label)

                            with open(os.path.join(root_dir, f'{data}_train_label{part_suffix}.pkl'), 'wb') as f:
                                pickle.dump(train_label, f)
                            with open(os.path.join(root_dir, f'{data}_val_label{part_suffix}.pkl'), 'wb') as f:
                                pickle.dump(val_label, f)
                            with open(os.path.join(root_dir, f'{data}_test_label{part_suffix}.pkl'), 'wb') as f:
                                pickle.dump(test_label, f)
                            with open(os.path.join(root_dir, f'{data}_train_index{part_suffix}.pkl'), 'wb') as f:
                                pickle.dump(idx_train, f)
                            with open(os.path.join(root_dir, f'{data}_val_index{part_suffix}.pkl'), 'wb') as f:
                                pickle.dump(idx_val, f)
                            with open(os.path.join(root_dir, f'{data}_test_index{part_suffix}.pkl'), 'wb') as f:
                                pickle.dump(idx_test, f)
                            with open(os.path.join(root_dir, f'{data}_all_label{part_suffix}.pkl'), 'wb') as f:
                                pickle.dump(labels, f)
                            with open(os.path.join(root_dir, f'{data}_features{part_suffix}.pkl'), 'wb') as f:
                                pickle.dump(features, f)

                            num_nodes = idx_train.numel()
                            n_perturbations = int(graph_noise * num_nodes)
                            print(n_perturbations)

                            train_label_np = train_label.detach().cpu().numpy() if isinstance(train_label, torch.Tensor) else np.asarray(train_label)
                            n_train_labels, _ = Noi_ab.noisify_with_P(train_label_np, nb_classes, label_noise, noise_type=noise_type)
                            n_train_labels_df = pd.DataFrame(n_train_labels)

                            with open(os.path.join(root_dir, f'{data}_train_label_new_{label_noise}{part_suffix}.pkl'), 'wb') as f:
                                pickle.dump(n_train_labels_df, f)
                            print(n_train_labels_df.head(3))
