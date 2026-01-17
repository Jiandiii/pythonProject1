import torch
# from DeepRobust.deeprobust.graph.global_attack import BaseAttack
from utils.base_attack import BaseAttack
# from torch.nn.parameter import Parameter
# from copy import deepcopy
# from DeepRobust.deeprobust.graph import utils
# import torch.nn.functional as F
import numpy as np
import random
from copy import deepcopy
import torch.nn.functional as F


class Random(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, add_nodes=False, device='cpu'):
        """
        """
        super(Random, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)

        self.add_nodes = add_nodes
        assert not self.attack_features, 'RND does NOT support attacking features'

    def attack(self, adj, n_perturbations,feats,g=0,l=0,data='Cora' ,type='add'):
        '''
        type: 'add', 'remove', 'flip','feature_based'
        '''
        if self.attack_structure:
            modified_adj = self.perturb_adj(adj, n_perturbations,feats,g,l,data=data, type=type)
            return modified_adj

    def perturb_adj(self, adj, n_perturbations,feats,g,l,data='Cora', type='add'):
        """
        Randomly add or flip edges.
        """
        # adj: sp.csr_matrix
        modified_adj = adj.tolil()

        type = type.lower()
        assert type in ['add', 'remove', 'flip','feature_based']

        if type == 'flip':
            # sample edges to flip
            edges = self.random_sample_edges(adj, n_perturbations, exclude=set())
            for n1, n2 in edges:
                modified_adj[n1, n2] = 1 - modified_adj[n1, n2]
                modified_adj[n2, n1] = 1 - modified_adj[n2, n1]
            self.modified_adj = modified_adj
            #self.save_adj(root=r'pro/{}/g_{}_l_{}/'.format(data, g, l), name='{}_mod_adj_flip_{}'.format(data, g))

        if type == 'add':
            if data=='ogbn-arxiv':
                nonzero = set(zip(*adj.nonzero()))
                edges = self.random_sample_edges(adj, n_perturbations, exclude=nonzero)
                for n1, n2 in edges:
                    modified_adj[n1, n2] = 1
                self.modified_adj = modified_adj
            # sample edges to add
            else:
                nonzero = set(zip(*adj.nonzero()))
                edges = self.random_sample_edges(adj, n_perturbations, exclude=nonzero)
                for n1, n2 in edges:
                    modified_adj[n1, n2] = 1
                    modified_adj[n2, n1] = 1
                self.modified_adj = modified_adj
            
            #self.save_adj(root=r'pro/{}/g_{}_l_{}/'.format(data, g, l), name='{}_mod_adj_add_{}'.format(data, g))

        if type == 'remove':
            # sample edges to remove
            nonzero = np.array(adj.nonzero()).T
            indices = np.random.permutation(nonzero)[: n_perturbations].T
            modified_adj[indices[0], indices[1]] = 0
            modified_adj[indices[1], indices[0]] = 0
            self.modified_adj = modified_adj
            #self.save_adj(root=r'pro/{}/g_{}_l_{}/'.format(data, g, l), name='{}_mod_adj_remove_{}'.format(data, g))

        if type == 'feature_based':
            # sample edges to remove

            nonzero = np.array(adj.nonzero()).T
            
            # feats = torch.tensor(feats, dtype=torch.float32, device='cuda:0')
            feats=feats.float()
            sim = F.normalize(feats, dim=1, p=2).mm(F.normalize(feats, dim=1, p=2).T).fill_diagonal_(0.0)
            sim[nonzero] = torch.inf
            flattened = sim.view(-1)  # 展平为一维张量
            values, indices = torch.topk(flattened, n_perturbations, largest=False)

            # 将一维索引转换为二维坐标
            rows = indices // sim.size(1)  # 行坐标
            cols = indices % sim.size(1)   # 列坐标
            # 组合为坐标列表
            coordinates = list(zip(rows.tolist(), cols.tolist()))
            # indices = np.random.permutation(nonzero)[: n_perturbations].T
            if data=='ogbn-arxiv':
                for n1, n2 in coordinates:
                    modified_adj[n1, n2] = 1
            else:
                for n1, n2 in coordinates:
                    modified_adj[n1, n2] = 1
                    modified_adj[n2, n1] = 1
            self.modified_adj = modified_adj

        # self.check_adj(modified_adj)
        return modified_adj

    # def perturb_features(self, features, n_perturbations):
    #     """
    #     Randomly perturb features.
    #     """
    #     print('number of pertubations: %s' % n_perturbations)
    #
    #     return modified_features

    def inject_nodes(self, adj, n_add, n_perturbations):
        """
        For each added node, randomly connect with other nodes.
        """
        # adj: sp.csr_matrix
        # TODO
        print('number of pertubations: %s' % n_perturbations)
        raise NotImplementedError

        modified_adj = adj.tolil()
        return modified_adj

    def random_sample_edges(self, adj, n, exclude):
        '''
            'exclude' is a set which contains the edges we do not want to sample
             and the edges already sampled
        '''
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

    def sample_forever(self, adj, exclude):
        '''
            'exclude' is a set which contains the edges we do not want to sample
             and the edges already sampled
        '''
        while True:
            # t = tuple(np.random.randint(0, adj.shape[0], 2))
            t = tuple(random.sample(range(0, adj.shape[0]), 2))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))

