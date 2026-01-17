import torch
from utils import process
from termcolor import cprint
import pickle
class embedder_single:
    def __init__(self, args,data,g,l):
        cprint("## Loading Dataset ##", "yellow")
        adj_list, features, labels, idx_train, idx_val, idx_test = process.load_single_graph(args)
        features = process.preprocess_features(features)

        args.nb_nodes = adj_list[0].shape[0]
        with open('data\{}\g_{}_l_{}\\nclass.pkl'.format(data,g,l,data), 'rb') as f:
            args.nb_classes  = pickle.load(f, encoding='latin1')
        with open('data\{}\g_{}_l_{}\dim.pkl'.format(data,g,l,data), 'rb') as f:
            args.ft_size  = pickle.load(f, encoding='latin1')


        self.adj_list = adj_list
        #self.dgl_graph = process.torch2dgl(adj_list[0])
        self.features = torch.FloatTensor(features)
        self.labels = labels.to(args.device)
        self.idx_train = idx_train.to(args.device)
        self.idx_val = idx_val.to(args.device)
        self.idx_test = idx_test.to(args.device)

        self.args = args


