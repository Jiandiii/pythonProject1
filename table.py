import pickle
import scipy
import argparse
from utils.load_noisydata import load_noisydata
from torch_geometric.utils import  dense_to_sparse
import numpy as np
import scipy.sparse as sp
import torch
import xlsxwriter as xw
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
parser.add_argument('--epochs', type=int, default=185,
                    help='training epochs')
args = parser.parse_args()
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
    idx_features_labels = np.genfromtxt("{}{}.content".format("./tmp/Cora/", "cora"), dtype=np.dtype(str))
    features2 = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    args.labels = torch.tensor(args.labels, dtype=torch.long, device='cuda:0')#clean label
    args.adj = torch.tensor(args.adj.toarray(), device='cuda:0')
    noise_label = args.labels.clone()
    noise_label[args.train_index] = args.train_label#noise label
    for i in range(args.epochs):
        with open('./label_analyze/epoch{}/score_label.pkl'.format(i), 'rb') as f:
            score = pickle.load(f, encoding='latin1')
        with open('./label_analyze/epoch{}/pseudo_label.pkl'.format(i), 'rb') as f:
            pseudo_label = pickle.load(f, encoding='latin1')
        with open('./label_analyze/epoch{}/loss_label.pkl'.format(i), 'rb') as f:
            loss = pickle.load(f, encoding='latin1')
        workbook = xw.Workbook('./label_analyze/table{}.xlsx'.format(i))  # 创建工作簿
        worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
        worksheet1.activate()
        title=['index','true_labels', 'noise_labels', 'predicted_label','loss','score']
        worksheet1.write_row('A1', title)
        t=1
        for j in args.train_index:
            worksheet1.write(t, 0, j)
            worksheet1.write(t, 1, args.labels[j])
            worksheet1.write(t, 2, noise_label[j])
            worksheet1.write(t, 3, pseudo_label[j])
            worksheet1.write(t, 4, loss[j])
            worksheet1.write(t, 5, score[j])
            print(j)
            t=t+1
        workbook.close()
        plt.hist(x=score.cpu().detach().numpy(), bins=100, density=True, alpha=0.7, color='skyblue',edgecolor='black', label='prediction')
        plt.title('label_analyze_predictions', fontsize=16)
        plt.xlabel('scores', fontsize=12)
        plt.ylabel('pdf', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('./label_analyze/{}.jpg'.format(i,i))
        plt.show()
if __name__ == "__main__":
    main()



