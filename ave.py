import pickle
import argparse
import torch
import numpy as np
from utils.load_noisydata import load_noisydata
parser = argparse.ArgumentParser(
    description="PyTorch implementation of model")
parser.add_argument('--dataset', type=str, default='Cora', help='dataset used.')
parser.add_argument('--noise_type', type=str, default='uniform', help='uniform,pair_noise,class_confusion,feature_based')
parser.add_argument('--teacher_student', type=str, default='student', help='teacher,student')
parser.add_argument('--epochs', type=int, default=5000,
                    help='number of epochs to train ')
parser.add_argument('--graph_noise', type=float, default=0.1,
                    help='rate of the graph_noise')
parser.add_argument('--label_noise', type=float, default=0.3,
                    help='rate of the label_noise')
parser.add_argument('--feature_noise', type=float, default=0.3, 
                    help='rate of the feature_noise')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    # print(preds)
    # print(preds.shape)
    labels_after = []
    for i in labels:
        labels_after.append(np.argmax(i.cpu().detach().numpy()))
    labels_after = torch.tensor(labels_after, device='cuda:0')
    correct = preds.eq(labels_after).double()
    correct = correct.sum()
    return correct / labels_after.shape[0]
def main():
    (args.train_index, args.val_index, args.test_index, args.train_label, args.val_label, args.test_label,
         args.features, args.adj, args.true_train_label, args.num_classes, args.labels) = load_noisydata(
            noise_type=args.noise_type,
            dataset=args.dataset,
            graph_noise=args.graph_noise,
            label_noise=args.label_noise,
            feature_noise=args.feature_noise
        )
    ones = torch.sparse.torch.eye(args.num_classes).to(args.device)
    args.targets_c_oneHot = ones.index_select(torch.tensor(0, device='cuda:0'),torch.tensor(args.labels, device='cuda:0'))
    path='./teacher_models/'+args.noise_type+args.dataset
    with open(path+'/g_{}_l_{}_f_{}/soft_label2020.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise), 'rb') as f:
        soft_label1 = pickle.load(f)
    with open(path+'/g_{}_l_{}_f_{}/soft_label2021.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise), 'rb') as f:
        soft_label2 = pickle.load(f)
    with open(path+'/g_{}_l_{}_f_{}/soft_label2022.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise), 'rb') as f:
        soft_label3 = pickle.load(f)
    with open(path+'/g_{}_l_{}_f_{}/soft_label2023.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise), 'rb') as f:
        soft_label4 = pickle.load(f)
    acc1=accuracy(soft_label1,args.targets_c_oneHot)
    acc2=accuracy(soft_label2,args.targets_c_oneHot)
    acc3=accuracy(soft_label3,args.targets_c_oneHot)
    acc4=accuracy(soft_label4,args.targets_c_oneHot)
    print((acc1+acc2+acc3+acc4)/4)
if __name__ == "__main__":
    main()
