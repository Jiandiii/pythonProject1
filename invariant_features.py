import networkx as nx
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from collections import defaultdict

# 1. 图结构预处理模块
class GraphProcessor:
    def __init__(self, n_hop=2, sample_ratio=0.3, label_key='label'):
        # super(GraphProcessor, self).__init__()
        self.n_hop = n_hop
        self.sample_ratio = sample_ratio
        self.label_key = label_key
        self.n_hop_neighbors = None
    
    def preprocess_graph(self, G):
        """预计算n跳同类别邻居"""
        self.n_hop_neighbors = defaultdict(list)
        
        for node in G.nodes():
            try:
                # 精确计算n跳邻居
                dist_dict = nx.single_source_shortest_path_length(G, node, cutoff=self.n_hop)
            except nx.NetworkXError:
                dist_dict = {}
            
            # 筛选恰好n跳的同类别节点
            valid_neighbors = [
                v for v, dist in dist_dict.items() 
                if dist == self.n_hop
                # if dist == self.n_hop and G.nodes[v]['label'] == G.nodes[node]['label']
            ]
            self.n_hop_neighbors[node] = valid_neighbors
        
        return self.n_hop_neighbors
    
    def generate_new_edges(self, node, num_samples=1):
        """生成多个采样后的边集合"""
        if node not in self.n_hop_neighbors:
            return []
        
        all_sampled = []
        neighbor_list = self.n_hop_neighbors[node]
        for _ in range(num_samples):
            if not neighbor_list:
                sampled = []
            else:
                # 确保至少采样1个节点
                k = max(1, int(len(neighbor_list) * self.sample_ratio))
                sampled = random.sample(neighbor_list, k)
            
            all_sampled.extend([[node, v] for v in sampled])
        
    
        return all_sampled
    
    def generate_edge_batch(self,nodes):
        """生成批量边索引"""
        edge_indices = []
        for node in nodes:
            edges = self.generate_new_edges(node)
            edge_indices.extend(edges)
        return [edge_indices]  # 简化为单批次

# 2. GCN模型定义（不含自环）
class PureNeighborGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # 定义无自环的GCN层
        
        self.gc1=GCNConv(input_dim, hidden_dim, add_self_loops=False)
        self.gc2=GCNConv(hidden_dim, output_dim, add_self_loops=False)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x, edge_indices):
        """处理多个采样图"""
        h = x.clone()
        h1=self.relu(self.gc1(h,edge_indices))
        h2=self.gc2(h1,edge_indices)
        return F.softmax(h2, dim=1)
    
    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
    # def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, self_loop=False, device=None):
    #     super(PureNeighborGCN, self).__init__()

    #     self.device = device
    #     self.gc1 = GCNConv(input_dim, hidden_dim, add_self_loops=self_loop)
    #     self.gc2 = GCNConv(hidden_dim, output_dim, add_self_loops=self_loop)
    #     self.dropout = dropout

    # def forward(self, x, edge_index):
    #     x1 = F.relu(self.gc1(x, edge_index),inplace=False)
    #     x1 = F.dropout(x1, self.dropout, training=self.training)
    #     x1 = self.gc2(x1, edge_index)
    #     # PyTorch内置函数
    #     # x1=x1*100
    #     # x1 = F.softmax(x1, dim=1)
    #     x1 = F.normalize(x1, p=2, dim=-1)
    #     return x1

    # def initialize(self):
    #     self.gc1.reset_parameters()
    #     self.gc2.reset_parameters()
class aggregate_nei(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # 定义无自环的GCN层
        
        self.linear=nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        """处理多个采样图"""
        h = self.linear(x)
        return h
