import pickle
import scipy
import argparse
from utils.load_noisydata import load_noisydata
from torch_geometric.utils import  dense_to_sparse
import numpy as np
import scipy.sparse as sp
import torch
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="PyTorch implementation of model")
parser.add_argument('--dataset', type=str, default='Cora', help='dataset used.')
parser.add_argument('--noise_type', type=str, default='uniform', help='uniform,pair_noise')
parser.add_argument('--graph_noise', type=float, default=0.3,
                    help='rate of the graph_noise')
parser.add_argument('--label_noise', type=float, default=0.3,
                    help='rate of the label_noise')
parser.add_argument('--feature_noise', type=float, default=0.3,
                    help='rate of the label_noise')
parser.add_argument('--epochs', type=int, default=198,
                    help='training epochs')
args = parser.parse_args()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def to_edge_index(adj):
    edge_index_all = dense_to_sparse(adj)
    edge_index = edge_index_all[0]
    edge_weight = edge_index_all[1]
    return edge_index, edge_weight

def main():
    (args.train_index, args.val_index, args.test_index, args.train_label, args.val_label, args.test_label,
             args.features, args.adj, args.true_train_label, args.num_classes, args.labels) = load_noisydata(
                noise_type=args.noise_type,
                dataset=args.dataset,
                graph_noise=args.graph_noise,
                label_noise=args.label_noise,
                feature_noise=args.feature_noise
            )
    args.train_label = args.train_label.values.reshape(-1)
    args.train_label = torch.from_numpy(args.train_label)
    args.train_label = torch.tensor(args.train_label, dtype=torch.long, device='cuda:0')

    args.train_index = torch.tensor(args.train_index, dtype=torch.long, device='cuda:0')
    args.train_index_clean = args.train_index.clone()
    args.val_index = torch.tensor(args.val_index, dtype=torch.long, device='cuda:0')
    args.test_index = torch.tensor(args.test_index, dtype=torch.long, device='cuda:0')
    args.ft_size = args.features.shape[1]
    args.features = torch.tensor(args.features, dtype=torch.float32, device='cuda:0')

    args.labels = torch.tensor(args.labels, dtype=torch.long, device='cuda:0')
    args.adj = torch.tensor(args.adj.toarray(), device='cuda:0')
    for label in range(7):
        if args.noise_type=='uniform':
            if args.label_noise !=0:
                with open('./loss_label_copl/loss_label_197.pkl', 'rb') as f:
                    losses = pickle.load(f, encoding='latin1')
            # if args.graph_noise !=0:
            #     with open('./loss_graph/loss_graph_{}.pkl'.format(epoch), 'rb') as f:
            #         losses = pickle.load(f, encoding='latin1')
        # if args.noise_type=='pair_noise':
        #     if args.label_noise !=0:
        #         with open('./loss_label/loss_label_pair_{}.pkl'.format(epoch), 'rb') as f:
        #             losses = pickle.load(f, encoding='latin1')
        #     if args.graph_noise !=0:
        #         with open('./loss_graph/loss_graph_pair_{}.pkl'.format(epoch), 'rb') as f:
        #             losses = pickle.load(f, encoding='latin1')
        label_ind=[]
        for i in range(len(args.labels)):
            if args.labels[i]==label:
                label_ind.append(i)

        losses=torch.tensor(np.array(losses), device='cuda:0')
        noise_label=args.labels.clone()
        noise_label[args.train_index]=args.train_label

        mod_lab_idx=noise_label!=args.labels
        losses=losses[label_ind]
        mod_lab_idx=mod_lab_idx[label_ind]


        idx_features_labels = np.genfromtxt("{}{}.content".format("./tmp/Cora/", "cora"), dtype=np.dtype(str))
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format("./tmp/Cora/", "cora"), dtype=np.int32)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
            edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(args.labels.shape[0], args.labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        A_I_nomal = normalize_adj(adj + sp.eye(adj.shape[0]))
        A_I_nomal = torch.tensor(A_I_nomal.toarray(), device='cuda:0')
        A_I_nomal= torch.where(A_I_nomal>0,1,0)
        adj=torch.where(args.adj>0,1,0)
        noisy_str_mask = A_I_nomal!= adj
        mod_edge_idx = noisy_str_mask.sum(1) > 0
        if args.label_noise !=0:
            mod_label=losses[mod_lab_idx]
            unmod_label=losses[mod_lab_idx==False]
            plt.hist(x=mod_label.cpu().detach().numpy(), bins=100, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='mod_label')
            plt.hist(x=unmod_label.cpu().detach().numpy(), bins=100, density=True, alpha=0.7, color='lightgreen', edgecolor='black', label='unmod_label')
            plt.title('label_influence_label{}'.format(label), fontsize=16)
            plt.xlabel('Loss', fontsize=12)
            plt.ylabel('pdf', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig('./analyze_copl/label{}.jpg'.format(label))
            plt.show()
        # if args.graph_noise !=0:
        #     mod_edge=losses[mod_edge_idx]
        #     unmod_edge=losses[mod_edge_idx==False]
        #     plt.hist(x=mod_edge.cpu().detach().numpy(), bins=100, density=True, alpha=0.7,  edgecolor='black', label='mod_edge')
        #     plt.hist(x=unmod_edge.cpu().detach().numpy(), bins=100, density=True, alpha=0.7, edgecolor='black',label='unmod_edge')
        #     plt.title('graph_influence', fontsize=16)
        #     plt.xlabel('Loss', fontsize=12)
        #     plt.ylabel('pdf', fontsize=12)
        #     plt.legend(fontsize=10)
        #     plt.grid(True, linestyle='--', alpha=0.7)
        #     plt.show()
        print(label)






if __name__ == "__main__":
    main()