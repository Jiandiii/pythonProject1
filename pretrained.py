import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import random
import scipy.sparse as sp
from utils.load_noisydata import load_noisydata
from models.gnn_models import GAT,GCN


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = F.softmax(model(F.normalize(features), adj))
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(features, adj)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test=accuracy(output[idx_test], labels[idx_test])
    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item(),acc_test


def compute_test():
    model.eval()
    output = model(F.normalize(features), adj)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--hidden', type=int, default=128, help='hidden size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--nheads', type=int, default=8, help='Number of head attentions')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=200, help='Patience')
    parser.add_argument('--seed', type=int, default=2024, help='Seed number')
    parser.add_argument('--graph_noise', type=float, default=0,
                        help='rate of the graph_noise')
    parser.add_argument('--label_noise', type=float, default=0.3,
                        help='rate of the label_noise')
    parser.add_argument('--dataset', type=str, default='Pubmed', help='dataset used.')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    (idx_train, idx_val, idx_test, train_label, args.val_label, args.test_label,
     features, adj, args.true_train_label, args.num_classes, labels) = load_noisydata(
        dataset=args.dataset,
        graph_noise=args.graph_noise,
        label_noise=args.label_noise
    )
    idx_train = torch.tensor(idx_train, dtype=torch.long, device='cuda:0')
    idx_val = torch.tensor(idx_val, dtype=torch.long, device='cuda:0')
    idx_test = torch.tensor(idx_test, dtype=torch.long, device='cuda:0')
    features = torch.tensor(features, dtype=torch.float32, device='cuda:0')
    labels = torch.tensor(labels, dtype=torch.long, device='cuda:0')
    adj = adj.tocoo()
    adj = torch.sparse.FloatTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                                   torch.FloatTensor(adj.data.astype(np.cfloat))).cuda()
    train_label = train_label.values.reshape(-1)
    train_label = torch.from_numpy(train_label)
    train_label = torch.tensor(train_label, dtype=torch.long, device='cuda:0')
    labels[idx_train]=train_label
    # model = GAT(input_size=features.shape[1], hidden_size=args.hidden, output_size=int(labels.max()) + 1,
    #             dropout=args.dropout, nheads=8, alpha=args.alpha).to(device='cuda:0')

    model=GCN(input_size=features.shape[1], hidden_size=args.hidden, output_size=int(labels.max()) + 1,dropout=args.dropout).to(device='cuda:0')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t_total = time.time()
    output_acc=[]
    loss_values = []
    bad_counter = 0
    best = 1000 + 1
    best_epoch = 0

    for epoch in range(1000):
        loss_val,acc_test=train(epoch)
        loss_values.append(loss_val)
        output_acc.append(acc_test.cpu().numpy())
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    output_acc = output_acc[-10:]
    output_acc = np.mean(output_acc)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("acc_test:{}".format(output_acc))
    compute_test()
    torch.save(model, './pretrained/{}_g{}_l{}.pth'.format(args.dataset, args.graph_noise, args.label_noise))


