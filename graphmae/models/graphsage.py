import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_geometric.nn import GCNConv,GATConv,SAGEConv,CuGraphSAGEConv
def remove_random_edges_undirected(edge_index, n_remove):
    num_edges = edge_index.size(1)
    
    if n_remove >= num_edges:
        raise ValueError("Cannot remove more edges than exist in the graph.")
    
    # 随机选择要删除的边的索引（确保成对删除）
    edges_to_remove = random.sample(range(num_edges // 2), n_remove )
    edges_to_remove = [i * 2 for i in edges_to_remove] + [i * 2 + 1 for i in edges_to_remove]
    
    mask = torch.ones(num_edges, dtype=torch.bool)
    mask[edges_to_remove] = False
    
    new_edge_index = edge_index[:, mask]
    return new_edge_index

class GraphSAGE(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(in_dim, num_hidden)
        self.sage2 = SAGEConv(num_hidden, out_dim)
        self.encoding=encoding
        
    def forward(self, g, inputs, return_hidden=False):
        hidden_list = []
        edge_index=torch.stack([g.edges()[0],g.edges()[1]])
        if self.encoding:
            n_remove=int(edge_index.shape[0]*0.3)
            edge_index=remove_random_edges_undirected(edge_index,n_remove)
        features = self.sage1(inputs, edge_index)
        hidden_list.append(features)
        features = F.relu(features)
        features = F.dropout(features, training=self.training)
        features = self.sage2(features, edge_index)
        hidden_list.append(features)
        if return_hidden:
            return F.log_softmax(features, dim=1), hidden_list
        else:
            return F.log_softmax(features, dim=1)
        # return F.log_softmax(features, dim=1)