from .base_attack import BaseAttack
import numpy as np
import random

class Gattack(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, add_nodes=False, device='cpu'):
        """
        """
        super(Gattack, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)

        self.add_nodes = add_nodes
        assert not self.attack_features, 'RND does NOT support attacking features'

    def attack(self, adj, n_perturbations, type='add',g=0,l=0,data='cora_after'):
        '''
        type: 'add', 'remove', 'flip'
        '''
        if self.attack_structure:
            modified_adj = self.perturb_adj(adj, n_perturbations, type,g,l,data)
            return modified_adj

    def perturb_adj(self, adj, n_perturbations, type='add',g=0,l=0,data='cora_after'):
        """
        Randomly add or flip edges.
        """
        # adj: sp.csr_matrix
        modified_adj = adj.tolil()

        type = type.lower()
        assert type in ['add', 'remove', 'flip']

        if type == 'flip':
            # sample edges to flip
            edges = self.random_sample_edges(adj, n_perturbations, exclude=set())
            for n1, n2 in edges:
                modified_adj[n1, n2] = 1 - modified_adj[n1, n2]
                modified_adj[n2, n1] = 1 - modified_adj[n2, n1]

        if type == 'add':
            # sample edges to add
            nonzero = set(zip(*adj.nonzero()))
            edges = self.random_sample_edges(adj, n_perturbations, exclude=nonzero)
            for n1, n2 in edges:
                modified_adj[n1, n2] = 1
                modified_adj[n2, n1] = 1
            self.modified_adj=modified_adj
            self.save_adj(root=r'{}/g_{}_l_{}/'.format(data,g,l), name='{}_mod_adj_add_{}'.format(data,g))
            #self.save_features(root=r'./data/', name='cora_mod_features_add_0.2')

        if type == 'remove':
            # sample edges to remove
            nonzero = np.array(adj.nonzero()).T
            ori=nonzero.shape[0]
            rem=nonzero.shape[0]-n_perturbations*2
            sh_del=int((ori-rem)/2)
            while sh_del !=0:
                print(nonzero)
                indices = np.random.permutation(nonzero)[: sh_del].T
                modified_adj[indices[0], indices[1]] = 0
                modified_adj[indices[1], indices[0]] = 0
                nonzero = np.array(adj.nonzero()).T
                ori = nonzero.shape[0]
                sh_del = int((ori - rem) / 2)
                print(sh_del)
            self.modified_adj = modified_adj
            self.save_adj(root=r'{}/g_{}_l_{}/'.format(data,g,l), name='{}_mod_adj_remove_{}'.format(data,g))
            #self.save_features(root=r'./data/', name='cora_mod_features_remove_0.2')





        self.check_adj(modified_adj)
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

