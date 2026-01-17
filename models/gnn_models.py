import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import random
import torch.nn.init as init
import dgl
import scipy.sparse as sp
import tensorflow as tf
class GATLayer(nn.Module):
    """GAT层"""

    def __init__(self, input_feature, output_feature, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = nn.Parameter(torch.empty(size=(2 * output_feature, 1)))
        self.w = nn.Parameter(torch.empty(size=(input_feature, output_feature)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.w)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)  # adj>0的位置使用e对应位置的值替换，其余都为-9e15，这样设定经过Softmax后每个节点对应的行非邻居都会变为0。
        attention = F.softmax(attention, dim=1)  # 每行做Softmax，相当于每个节点做softmax
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.mm(attention, Wh)  # 得到下一层的输入

        if self.concat:
            return F.elu(h_prime)  # 激活
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.output_feature, :])  # N*out_size @ out_size*1 = N*1

        Wh2 = torch.matmul(Wh, self.a[self.output_feature:, :])  # N*1

        e = Wh1 + Wh2.T  # Wh1的每个原始与Wh2的所有元素相加，生成N*N的矩阵
        return self.leakyrelu(e)


class GAT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, alpha, nheads, concat=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attention = [GATLayer(input_size, hidden_size, dropout=dropout, alpha=alpha, concat=True) for _ in
                          range(nheads)]
        for i, attention in enumerate(self.attention):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(hidden_size * nheads, output_size, dropout=dropout, alpha=alpha, concat=False)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attention], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        #return F.log_softmax(x, dim=1)
        return x


# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=8, att_dropout=0):
#         super().__init__()
#         self.convs = ModuleList()
#         self.convs.append(GATConv(in_channels, hidden_channels//heads, heads=heads, dropout=att_dropout))
#         for i in range(0, num_layers-2):
#             self.convs.append(GATConv(hidden_channels, hidden_channels//heads, heads=heads, dropout=att_dropout))
#         self.convs.append(GATConv(hidden_channels, out_channels, dropout=att_dropout))
#
#     def encode(self, x, edge_index):
#         for conv in self.convs[:-1]:
#             x = conv(x, edge_index).relu()
#         x = self.convs[-1](x, edge_index)
#         return x
#
#     def decode(self, z, edge_label_index):
#         hidden = z[edge_label_index[0]] * z[edge_label_index[1]]
#         logits = (hidden).sum(dim=-1)
#         hidden = F.normalize(hidden, dim=1)
#         return hidden, logits
#
#     def decode_all(self, z):
#         prob_adj = z @ z.t()
#         return (prob_adj > 0).nonzero(as_tuple=False).t()


class GCNLayer(nn.Module):

    def __init__(self,input_features,output_features,bias=False):
        super(GCNLayer,self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weights = nn.Parameter(torch.FloatTensor(input_features,output_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):

        std = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std,std)
        if self.bias is not None:
            self.bias.data.uniform_(-std,std)

    def forward(self,adj,x):
        support = torch.mm(x,self.weights)
        output = torch.spmm(adj,support)
        if self.bias is not None:
            return output+self.bias
        return output

class GCN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,dropout,bias=False):
        super(GCN,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_class = output_size
        self.gcn1 = GCNLayer(input_size,hidden_size,bias=bias)
        self.gcn2 = GCNLayer(hidden_size,output_size,bias=bias)
        self.dropout = dropout
    def forward(self,x,adj):
        x = F.relu(self.gcn1(adj,x))
        x = F.dropout(x,self.dropout,training=self.training)
        x = self.gcn2(adj,x)
        return x
# class GCN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, dropout=0.5, self_loop=True, device=None):
#
#         super(GCN, self).__init__()
#
#         self.device = device
#         self.gc1 = GCNConv(input_size, hidden_size, bias=True, add_self_loops=self_loop)
#         self.gc2 = GCNConv(hidden_size, output_size, bias=True, add_self_loops=self_loop)
#         self.dropout = dropout
#
#     def forward(self, x, edge_index, edge_weight):
#         x1 = F.relu(self.gc1(x, edge_index, edge_weight))
#         x1 = F.dropout(x1, self.dropout, training=self.training)
#         x1 = self.gc2(x1, edge_index, edge_weight)
#         return x1
#
#     def initialize(self):
#         self.gc1.reset_parameters()
#         self.gc2.reset_parameters()

class GraphSAGE(nn.Module):
    """https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_full.py"""
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        from dgl.nn.pytorch.conv import SAGEConv

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, inputs, graph):
        graph=graph.to_dense().cpu().numpy()
        # inputs=inputs.cpu().numpy()
        graph=sp.coo_matrix(graph)
        graph=dgl.from_scipy(graph)
        graph = graph.to(device='cuda:0')

        h = self.dropout(inputs)
        # h = h.cpu()
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class GCN_LP(torch.nn.Module):
    def __init__(self, in_channels,hidden_size, out_channels):
        super(GCN_LP, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_size)
        self.conv2 = GCNConv(hidden_size, out_channels)
        self.w=nn.Parameter(torch.randn(out_channels))


    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()




