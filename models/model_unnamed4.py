import filecmp
import dgl.function as fn
import torch.optim as optim
from torch_geometric.nn import MessagePassing, GCNConv
from torch_scatter import scatter
from models.encoder_decoder import *
from scipy.sparse import coo_matrix
from models.Semi_RNCGLN import *
from torch_geometric.utils import (to_undirected, dense_to_sparse, to_dense_adj, 
                                   is_undirected, remove_self_loops, degree, 
                                   to_scipy_sparse_matrix, from_scipy_sparse_matrix,
                                   negative_sampling, coalesce)
import torch_geometric.utils as tu
from torch.autograd import Variable
import matplotlib.pyplot as plt
from pretrained import train
from utils import loss
from utils import losses
import torch_geometric
from copy import deepcopy
import networkx as nx
from graphmae.models import build_model
from invariant_features import *
import xlsxwriter as xw
import torch.nn.functional as F
import numpy as np
import random
import time
import pickle
import os
import torch
import faiss
from collections import defaultdict

# def get_top_k_similar_edge_index(features, global_top_k=100, local_top_k=200):
#     print("开始使用FAISS构建Top-K相似边索引...")
#     normalized_feat = F.normalize(features, dim=1, p=2).cpu().numpy().astype(np.float32)
#     N, D = normalized_feat.shape
    
#     # 关键：使用GPU索引
#     print("构建FAISS GPU索引...")
#     res = faiss.StandardGpuResources()  # 分配GPU资源
#     index_flat = faiss.IndexFlatIP(D)
#     # 将CPU索引转为GPU索引（0表示第0块GPU）
#     index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    
#     index.add(normalized_feat)
#     similarities, indices = index.search(normalized_feat, local_top_k)  # GPU加速搜索
#     print("FAISS搜索完成，开始收集候选点对...")
#     # 3. 收集候选点对，去重并排除自环（u < v确保无序去重）
#     pair_sim = defaultdict(float)
#     for u in range(N):
#         for i in range(local_top_k):
#             v = indices[u][i]
#             sim = similarities[u][i]
#             if u == v:  # 排除自环
#                 continue
#             if u > v:  # 确保u < v，避免(u,v)和(v,u)重复
#                 u, v = v, u
#             if sim > pair_sim[(u, v)]:
#                 pair_sim[(u, v)] = sim
#     print("候选点对收集完成，开始排序并选择Top-K...")
#     # 4. 按相似度降序排序，取前global_top_k个点对
#     sorted_pairs = sorted(pair_sim.items(), key=lambda x: x[1], reverse=True)
#     top_pairs = [p[0] for p in sorted_pairs[:global_top_k]]  # 只保留(u, v)，忽略相似度
#     print("全局Top-K点对选择完成，开始转换为edge_index格式...")
#     # 5. 转换为edge_index格式（[2, E]长整型张量）
#     u_list = [p[0] for p in top_pairs]
#     v_list = [p[1] for p in top_pairs]
#     edge_index = torch.tensor([u_list, v_list], dtype=torch.long, device=features.device)
#     print("edge_index转换完成，形状:", edge_index.shape)
#     return edge_index
def get_top_k_similar_edge_index(features, K=10):
    """
    GPU版本：为每个节点找到Top-K相似节点，生成无向边索引（[2, E]）
    参数：
        features: 节点特征矩阵（GPU张量，形状[N, D]）
        K: 每个节点的近邻数量
    返回：
        无向化的边索引（GPU张量）
    """
    # 1. 特征归一化（在GPU上执行）
    print("开始使用FAISS GPU版构建Top-K相似边索引...")
    normalized_feat = F.normalize(features, dim=1, p=2)  # 保持在GPU上
    
    # 2. 准备FAISS GPU资源（分配GPU内存）
    print("构建FAISS GPU索引...")
    res = faiss.StandardGpuResources()  # 管理FAISS的GPU资源
    
    # 3. 构建GPU版内积索引（等价于余弦相似度，因特征已归一化）
    D = normalized_feat.shape[1]  # 特征维度
    cpu_index = faiss.IndexFlatIP(D)  # CPU基础索引
    # 将CPU索引转换为GPU索引（0表示使用第0块GPU）
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    
    # 4. 将归一化特征从GPU张量转为CPU numpy数组（FAISS需numpy输入，内部会传到GPU）
    # 注意：这一步是临时传输，搜索逻辑仍在GPU上执行
    normalized_feat_np = normalized_feat.cpu().numpy().astype(np.float32)
    
    # 5. 在GPU上添加特征并搜索Top-K+1近邻（+1是为了后续排除自环）
    gpu_index.add(normalized_feat_np)  # 特征传入GPU索引
    _, indices = gpu_index.search(normalized_feat_np, K + 1)  # 搜索结果（GPU加速）
    
    # 6. 排除自环，收集边的源节点（src）和目标节点（dst）
    src = []
    dst = []
    N = features.shape[0]  # 节点总数
    for u in range(N):
        # 遍历当前节点u的Top-K+1近邻
        for v in indices[u]:
            if u != v:  # 跳过自环（u == v）
                src.append(u)
                dst.append(v)
    
    # 7. 转换为GPU张量的边索引格式，并转为无向图
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=features.device)
    print("edge_index转换完成，形状:", edge_index.shape)
    return tu.to_undirected(edge_index)  # 无向化（与原逻辑一致）
# ===================== 全局变量：时间统计字典 =====================
time_stats = {}  # 存储每个操作的耗时，key: 操作标识，value: 耗时（秒）
current_timers = {}  # 存储当前正在进行的操作的开始时间

# ===================== 资源监控函数（合并时间统计+显存监控） =====================
def monitor_resource(operation="", prefix="", mode="start"):
    """
    合并时间统计和显存监控的统一函数
    Args:
        operation: 当前执行的操作描述（需唯一标识）
        prefix: 前缀标识（如"PRE"="开始前"，"POST"="完成后"）
        mode: "start"=操作开始（记录开始时间+打印显存），"end"=操作结束（计算耗时+打印显存+记录时间）
    """
    # 生成唯一操作标识（避免重复）
    op_key = f"{prefix}_{operation}" if prefix else operation
    
    # 打印显存使用情况
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated("cuda:0") / (1024 ** 2)
        reserved = torch.cuda.memory_reserved("cuda:0") / (1024 ** 2)
        # 格式化显存单位
        allocated_str = f"{allocated / 1024:.2f} GB" if allocated > 1024 else f"{allocated:.2f} MB"
        reserved_str = f"{reserved / 1024:.2f} GB" if reserved > 1024 else f"{reserved:.2f} MB"
    else:
        allocated_str = "N/A"
        reserved_str = "N/A"
    
    # 时间统计逻辑
    if mode == "start":
        # 记录开始时间
        start_time = time.perf_counter()
        current_timers[op_key] = start_time
        # 打印开始信息（显存+开始时间）
        print(f"[{prefix}] {operation} | 开始时间: {time.strftime('%H:%M:%S', time.localtime(start_time))} | 已分配显存: {allocated_str} | 预留显存: {reserved_str}")
    
    elif mode == "end":
        # 计算耗时
        if op_key not in current_timers:
            print(f"警告：操作 {op_key} 未记录开始时间，跳过时间统计")
            return
        end_time = time.perf_counter()
        elapsed = end_time - current_timers[op_key]
        # 保存耗时到统计字典
        time_stats[op_key] = elapsed
        # 打印结束信息（显存+结束时间+耗时）
        print(f"[{prefix}] {operation} | 结束时间: {time.strftime('%H:%M:%S', time.localtime(end_time))} | 耗时: {elapsed:.4f}s | 已分配显存: {allocated_str} | 预留显存: {reserved_str}")
        # 删除当前计时器
        del current_timers[op_key]

# ===================== 初始化监控（首次GPU使用前） =====================
# monitor_resource(operation="代码启动，未使用GPU", prefix="INIT", mode="start")
# monitor_resource(operation="代码启动，未使用GPU", prefix="INIT", mode="end")

def tensor_intersection(tensor1, tensor2):
    # monitor_resource(operation="tensor_intersection - 开始计算张量交集", prefix="PRE", mode="start")
    mask = torch.isin(tensor1, tensor2)
    intersection = tensor1[mask]
    result = torch.unique(intersection)
    # monitor_resource(operation="tensor_intersection - 完成张量交集计算", prefix="POST", mode="end")
    return result

def calculate_classification_metrics(logits, targets_onehot, num_classes=None, average='macro'):
    # monitor_resource(operation="calculate_classification_metrics - 开始计算分类指标", prefix="PRE", mode="start")
    """
    计算多分类任务的精确率、召回率和F1分数
    
    参数:
        logits: 模型输出的logits，形状为 (batch_size, num_classes)
        targets_onehot: 真实标签的one-hot表示，形状为 (batch_size, num_classes)
        num_classes: 类别数量，默认为None(自动从数据中推断)
        average: 计算方式，可选 'macro'、'micro'、'weighted' 或 None(返回各类别指标)
    
    返回:
        precision: 精确率
        recall: 召回率
        f1_score: F1分数
    """
    # 将logits转换为预测类别索引
    # monitor_resource(operation="calculate_classification_metrics - 完成预测索引转换", prefix="MID1", mode="start")
    preds_indices = logits.argmax(dim=1)
    # monitor_resource(operation="calculate_classification_metrics - 完成预测索引转换", prefix="MID1", mode="end")
    
    # 将one-hot标签转换为类别索引
    # monitor_resource(operation="calculate_classification_metrics - 完成真实标签索引转换", prefix="MID2", mode="start")
    targets_indices = targets_onehot.argmax(dim=1)
    # monitor_resource(operation="calculate_classification_metrics - 完成真实标签索引转换", prefix="MID2", mode="end")
    
    # 自动推断类别数
    if num_classes is None:
        num_classes = max(logits.shape[1], targets_indices.max().item() + 1)
    
    # 计算混淆矩阵元素
    # monitor_resource(operation="calculate_classification_metrics - 初始化混淆矩阵张量", prefix="MID3", mode="start")
    tp = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    fp = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    fn = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    # monitor_resource(operation="calculate_classification_metrics - 初始化混淆矩阵张量", prefix="MID3", mode="end")
    
    for c in range(num_classes):
        # 当前类别的预测和真实标签
        preds_c = (preds_indices == c)
        targets_c = (targets_indices == c)
        
        tp[c] = (preds_c & targets_c).sum()
        fp[c] = (preds_c & (~targets_c)).sum()
        fn[c] = ((~preds_c) & targets_c).sum()
    # monitor_resource(operation="calculate_classification_metrics - 完成混淆矩阵计算", prefix="MID4", mode="start")
    # monitor_resource(operation="calculate_classification_metrics - 完成混淆矩阵计算", prefix="MID4", mode="end")
    
    # 计算各类别的精确率、召回率和F1分数
    # monitor_resource(operation="calculate_classification_metrics - 完成类别级指标计算", prefix="MID5", mode="start")
    precision_per_class = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    recall_per_class = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    f1_per_class = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    
    # 避免除零错误
    mask = (tp + fp) > 0
    precision_per_class[mask] = tp[mask] / (tp[mask] + fp[mask])
    
    mask = (tp + fn) > 0
    recall_per_class[mask] = tp[mask] / (tp[mask] + fn[mask])
    
    mask = (precision_per_class + recall_per_class) > 0
    f1_per_class[mask] = 2 * (precision_per_class[mask] * recall_per_class[mask]) / (precision_per_class[mask] + recall_per_class[mask])
    # monitor_resource(operation="calculate_classification_metrics - 完成类别级指标计算", prefix="MID5", mode="end")
    
    # 根据average参数聚合结果
    if average == 'macro':
        # 计算宏平均（各类别平等权重）
        precision = precision_per_class.mean()
        recall = recall_per_class.mean()
        f1_score = f1_per_class.mean()
    elif average == 'micro':
        # 计算微平均（全局TP、FP、FN）
        total_tp = tp.sum()
        total_fp = fp.sum()
        total_fn = fn.sum()
        
        if total_tp + total_fp > 0:
            precision = total_tp / (total_tp + total_fp)
        else:
            precision = torch.tensor(0.0, device=logits.device)
        
        if total_tp + total_fn > 0:
            recall = total_tp / (total_tp + total_fn)
        else:
            recall = torch.tensor(0.0, device=logits.device)
        
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = torch.tensor(0.0, device=logits.device)
    elif average == 'weighted':
        # 计算加权平均（按类别样本数加权）
        class_weights = torch.bincount(targets_indices, minlength=num_classes).float()
        total_samples = class_weights.sum()
        
        if total_samples > 0:
            weights = class_weights / total_samples
            precision = (precision_per_class * weights).sum()
            recall = (recall_per_class * weights).sum()
            f1_score = (f1_per_class * weights).sum()
        else:
            precision = torch.tensor(0.0, device=logits.device)
            recall = torch.tensor(0.0, device=logits.device)
            f1_score = torch.tensor(0.0, device=logits.device)
    else:
        # 返回各类别的指标
        precision = precision_per_class
        recall = recall_per_class
        f1_score = f1_per_class
    
    # monitor_resource(operation="calculate_classification_metrics - 完成指标聚合，函数结束", prefix="POST", mode="end")
    return precision, recall, f1_score


# class DimWiseFusion(nn.Module):
#     def __init__(self, dim):
        
#         # monitor_resource(operation="DimWiseFusion - 初始化前", prefix="PRE", mode="start")
#         super(DimWiseFusion, self).__init__()
#         # 初始化与特征维度相同的可学习权重w
#         self.w = nn.Parameter(torch.randn(dim))
#         # 使用sigmoid确保每个维度的w在[0,1]范围内
#         # monitor_resource(operation="DimWiseFusion - 初始化完成（创建可学习权重w）", prefix="POST", mode="end")
        
#     def forward(self, A0, An, ver):
#         # monitor_resource(operation="DimWiseFusion.forward - 特征融合前", prefix="PRE", mode="start")
#         w = torch.sigmoid(self.w)  # 每个维度的权重独立控制在[0,1]
#         if ver == 1:
#             w = (w > 0.5).float()
#         result = w * A0 + (1 - w) * An  # 点乘融合
#         #显存增加较多
#         # monitor_resource(operation="DimWiseFusion.forward - 特征融合完成", prefix="POST", mode="end")
#         return result

class DimWiseFusion(nn.Module):
    def __init__(self):  # 移除dim参数，无需维度相关权重
        # monitor_resource(operation="DimWiseFusion - 初始化前", prefix="PRE", mode="start")
        super(DimWiseFusion, self).__init__()
        # 可学习的概率参数：控制保留A0边的概率（初始值设为0.5）
        self.p = nn.Parameter(torch.tensor(0.1))  # 仅1个参数，大幅减少参数量
        # monitor_resource(operation="DimWiseFusion - 初始化完成（创建概率参数p）", prefix="POST", mode="end")
        
    def forward(self, A0_edge_index, An_edge_index, ver):
        # monitor_resource(operation="DimWiseFusion.forward - 边融合前", prefix="PRE", mode="start")
        
        if ver == 1:
            # 1. 计算保留A0边的概率（转为标量数值）
            keep_prob = torch.sigmoid(self.p)  # 此时是0维Tensor
            keep_prob_val = keep_prob.item()   # 关键修改：转为Python标量（float）
            
            # 2. 对A0的边按概率keep_prob_val采样
            num_A0_edges = A0_edge_index.shape[1]
            if num_A0_edges == 0:
                fused_edge_index = An_edge_index
            else:
                # 生成采样掩码：fill_value用标量keep_prob_val，设备与A0_edge_index一致
                mask = torch.bernoulli(
                    torch.full(
                        (num_A0_edges,),  # 形状：边的数量
                        keep_prob_val,    # 标量数值（解决类型错误）
                        device=A0_edge_index.device  # 设备指定
                    )
                )
                mask = mask.bool()
                A0_kept = A0_edge_index[:, mask]
                
                # 3. 合并并去重
                combined = torch.cat([A0_kept, An_edge_index], dim=1)
                fused_edge_index = torch.unique(combined, dim=1)
        
        else:
            # 其他版本的逻辑（根据需求调整）
            w = torch.sigmoid(self.p).item()
            fused_edge_index = w * A0_edge_index + (1 - w) * An_edge_index  # 若为特征矩阵
            
        # monitor_resource(operation="DimWiseFusion.forward - 边融合完成", prefix="POST", mode="end")
        return fused_edge_index

def adjacency_ce_loss(adj1, adj2):
    # monitor_resource(operation="adjacency_ce_loss - 计算邻接矩阵交叉熵前", prefix="PRE", mode="start")
    """
    计算两个邻接矩阵之间的交叉熵损失
    
    参数:
        adj1 (torch.Tensor): 第一个邻接矩阵(概率)
        adj2 (torch.Tensor): 第二个邻接矩阵(0/1)
        
    返回:
        torch.Tensor: 交叉熵损失值
    """
    ce_loss = nn.BCELoss()
    loss = ce_loss(adj1, adj2)
    # monitor_resource(operation="adjacency_ce_loss - 交叉熵损失计算完成", prefix="POST", mode="end")
    return loss

def remove_duplicate_edges(edge_index, edge_weights):
    # monitor_resource(operation="remove_duplicate_edges - 去重前", prefix="PRE", mode="start")
    # 转置 edge_index 并拼接权重，便于去重
    edges_with_weights = torch.cat([
        edge_index.t(),
        edge_weights.unsqueeze(1)
    ], dim=1)  # shape: [num_edges, 3] (u, v, weight)
    # monitor_resource(operation="remove_duplicate_edges - 拼接边与权重完成", prefix="MID1", mode="start")
    # monitor_resource(operation="remove_duplicate_edges - 拼接边与权重完成", prefix="MID1", mode="end")

    # 对 (u, v) 去重，保留第一个出现的边
    unique_edges, indices = torch.unique(
        edges_with_weights[:, :2],  # 仅针对 (u, v) 去重
        dim=0,
        return_inverse=True,
        return_counts=False
    )
    # monitor_resource(operation="remove_duplicate_edges - 边去重完成", prefix="MID2", mode="start")
    # monitor_resource(operation="remove_duplicate_edges - 边去重完成", prefix="MID2", mode="end")

    # 获取去重后的权重（取每组重复边的第一个权重）
    unique_weights = edge_weights[indices.unique()]

    # 转置回 [2, num_unique_edges] 格式
    unique_edge_index = unique_edges.t().to(torch.long)
    # monitor_resource(operation="remove_duplicate_edges - 格式转换完成，函数结束", prefix="POST", mode="end")

    return unique_edge_index, unique_weights

class WeightedAggregation(MessagePassing):
    def __init__(self):
        # monitor_resource(operation="WeightedAggregation - 初始化前", prefix="PRE", mode="start")
        super().__init__(aggr='add')  # 初始聚合方式设为求和
        # monitor_resource(operation="WeightedAggregation - 初始化完成", prefix="POST", mode="end")
    
    def forward(self, x, edge_index, edge_weight):
        # monitor_resource(operation="WeightedAggregation.forward - 消息传递前", prefix="PRE", mode="start")
        # 移除自环边
        # monitor_resource(operation="WeightedAggregation.forward - 移除自环边完成", prefix="MID1", mode="start")
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        # monitor_resource(operation="WeightedAggregation.forward - 移除自环边完成", prefix="MID1", mode="end")
        
        # 计算归一化权重（可选）
        # monitor_resource(operation="WeightedAggregation.forward - 权重归一化完成", prefix="MID2", mode="start")
        row, col = edge_index
        deg = torch.zeros(x.size(0), device=x.device)
        deg.scatter_add_(0, col, edge_weight)  # 按目标节点累加权重
        deg_inv = 1.0 / deg
        deg_inv[deg == 0] = 0  # 处理无邻居节点
        norm_weight = edge_weight * deg_inv[col]  # 归一化权重
        # monitor_resource(operation="WeightedAggregation.forward - 权重归一化完成", prefix="MID2", mode="end")
        
        # 消息传递（加权平均）
        result = self.propagate(edge_index, x=x, norm_weight=norm_weight)
        # monitor_resource(operation="WeightedAggregation.forward - 消息传递完成，函数结束", prefix="POST", mode="end")
        return result
    
    def message(self, x_j, norm_weight):
        # monitor_resource(operation="WeightedAggregation.message - 生成消息前", prefix="PRE", mode="start")
        message = x_j * norm_weight.view(-1, 1)  # 消息=邻居特征×归一化权重
        # monitor_resource(operation="WeightedAggregation.message - 消息生成完成", prefix="POST", mode="end")
        return message

def remove_random_edges_undirected(edge_index, n_remove):
    # monitor_resource(operation="remove_random_edges_undirected - 随机删边前", prefix="PRE", mode="start")
    num_edges = edge_index.size(1)
    
    if n_remove >= num_edges:
        raise ValueError("Cannot remove more edges than exist in the graph.")
    
    # 随机选择要删除的边的索引（确保成对删除）
    edges_to_remove = random.sample(range(num_edges // 2), n_remove )
    edges_to_remove = [i * 2 for i in edges_to_remove] + [i * 2 + 1 for i in edges_to_remove]
    
    mask = torch.ones(num_edges, dtype=torch.bool)
    mask[edges_to_remove] = False
    
    new_edge_index = edge_index[:, mask]
    # monitor_resource(operation="remove_random_edges_undirected - 随机删边完成", prefix="POST", mode="end")
    return new_edge_index

def get_A_r(adj, r):
    # monitor_resource(operation="get_A_r - 计算r阶邻接矩阵前", prefix="PRE", mode="start")
    adj_label = adj
    for i in range(r - 1):
        # monitor_resource(operation=f"get_A_r - 第{i+1}次矩阵乘法开始", prefix=f"MID_{i+1}", mode="start")
        adj_label = adj_label @ adj  # 矩阵乘法（高显存消耗）
        # monitor_resource(operation=f"get_A_r - 第{i+1}次矩阵乘法完成", prefix=f"MID_{i+1}", mode="end")
    # monitor_resource(operation="get_A_r - r阶邻接矩阵计算完成", prefix="POST", mode="end")
    return adj_label

def to_edge_set(edge_index, directed=False):
    # monitor_resource(operation="to_edge_set - 转换为边集合前", prefix="PRE", mode="start")
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    edges = set()
    for s, d in zip(src, dst):
        if not directed:  # 无向图时排序节点对
            s, d = (s, d) if s <= d else (d, s)
        edges.add((s, d))
    # monitor_resource(operation="to_edge_set - 边集合转换完成", prefix="POST", mode="end")
    return edges

def remove_edges_adj_matrix(adj_matrix, n_remove):
    # monitor_resource(operation="remove_edges_adj_matrix - 邻接矩阵删边前", prefix="PRE", mode="start")
    edge_indices = np.argwhere(adj_matrix > 0)

    if len(edge_indices) < n_remove:
        raise ValueError("Cannot remove more edges than exist in the graph.")

    # 随机选择要删除的边
    edges_to_remove = random.sample(list(edge_indices), n_remove)

    # 在邻接矩阵中置 0
    new_adj = adj_matrix.copy()
    for edge in edges_to_remove:
        i, j = edge
        new_adj[i, j] = 0
        if is_undirected:
            new_adj[j, i] = 0  # 无向图需对称置零
    # monitor_resource(operation="remove_edges_adj_matrix - 邻接矩阵删边完成", prefix="POST", mode="end")
    return new_adj

# def to_edge_index(adj):
#     monitor_resource(operation="to_edge_index - 邻接矩阵转边索引前", prefix="PRE", mode="start")
#     edge_index_all = dense_to_sparse(adj)  # 高显存消耗（稠密矩阵转稀疏）
#     edge_index = edge_index_all[0]
#     edge_weight = edge_index_all[1]
#     monitor_resource(operation="to_edge_index - 邻接矩阵转边索引完成", prefix="POST", mode="end")
#     return edge_index, edge_weight
def to_edge_index(adj):  # 传入的adj是scipy.coo_matrix
    # monitor_resource(operation="to_edge_index - 邻接矩阵转边索引前", prefix="PRE", mode="start")
    
    # 直接从COO矩阵提取稀疏信息（无任何稠密步骤）
    row = torch.from_numpy(adj.row).long().to('cuda:0')  # 源节点索引（long类型，转GPU）
    col = torch.from_numpy(adj.col).long().to('cuda:0')  # 目标节点索引（long类型，转GPU）
    edge_index = torch.stack([row, col], dim=0)  # 边索引 shape: [2, E]（E是边数）
    
    # 提取边权重（COO矩阵的data属性）
    edge_weight = torch.from_numpy(adj.data).float().to('cuda:0')  # 边权重 shape: [E]
    
    # monitor_resource(operation="to_edge_index - 邻接矩阵转边索引完成", prefix="POST", mode="end")
    return edge_index, edge_weight

def to_adj(edge_index, edge_weight, x):
    # monitor_resource(operation="to_adj - 边索引转邻接矩阵前", prefix="PRE", mode="start")
    adj = coo_matrix((edge_weight.cpu().detach().numpy(), edge_index.cpu().detach().numpy()), shape=(x, x))
    adj = torch.tensor(adj.toarray(), device='cuda:0', dtype=torch.float32)  # CPU→GPU，显存增加
    #显存增加较多
    # monitor_resource(operation="to_adj - 边索引转邻接矩阵完成（已转移到GPU）", prefix="POST", mode="end")
    return adj

def cosine_error_loss(x, y, alpha=1):
    # monitor_resource(operation="cosine_error_loss - 计算余弦误差损失前", prefix="PRE", mode="start")
    # x = F.normalize(x, p=2, dim=-1)
    # x = F.softmax(x, dim=1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    # loss=torch.exp((x * y).sum(dim=-1)-1)
    loss = loss.mean()
    # monitor_resource(operation="cosine_error_loss - 余弦误差损失计算完成", prefix="POST", mode="end")
    return loss

def accuracy(output, labels):
    # monitor_resource(operation="accuracy - 计算准确率前", prefix="PRE", mode="start")
    preds = output.max(1)[1].type_as(labels)
    # print(preds)
    # print(preds.shape)
    labels_after = []
    for i in labels:
        labels_after.append(np.argmax(i.cpu().detach().numpy()))
    labels_after = torch.tensor(labels_after, device='cuda:0')  # CPU→GPU
    correct = preds.eq(labels_after).double()
    correct = correct.sum()
    acc = correct / labels_after.shape[0]
    # monitor_resource(operation="accuracy - 准确率计算完成", prefix="POST", mode="end")
    return acc

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, self_loop=True, device=None):
        # monitor_resource(operation="GCN - 初始化前", prefix="PRE", mode="start")
        super(GCN, self).__init__()

        self.device = device
        self.gc1 = GCNConv(nfeat, nhid, bias=True, add_self_loops=self_loop)  # 第一层GCN
        self.gc2 = GCNConv(nhid, nclass, bias=True, add_self_loops=self_loop)  # 第二层GCN
        self.dropout = dropout
        # monitor_resource(operation="GCN - 初始化完成（创建2层GCNConv）", prefix="POST", mode="end")

    def forward(self, x, edge_index, edge_weight):
        # monitor_resource(operation="GCN.forward - 前向传播前", prefix="PRE", mode="start")
        # 第一层GCN
        # monitor_resource(operation="GCN.forward - 第一层GCN完成", prefix="MID1", mode="start")
        x1 = F.relu(self.gc1(x, edge_index, edge_weight))  # 第一层GCN+激活（高显存）
        # monitor_resource(operation="GCN.forward - 第一层GCN完成", prefix="MID1", mode="end")
        x1 = F.dropout(x1, self.dropout, training=self.training)  # Dropout
        # 第二层GCN
        # monitor_resource(operation="GCN.forward - 第二层GCN完成", prefix="MID2", mode="start")
        x1 = self.gc2(x1, edge_index, edge_weight)  # 第二层GCN（高显存）
        # monitor_resource(operation="GCN.forward - 第二层GCN完成", prefix="MID2", mode="end")
        # PyTorch内置函数
        # x1=x1*100
        # x1 = F.softmax(x1, dim=1)
        # monitor_resource(operation="GCN.forward - 前向传播完成", prefix="POST", mode="start")
        x1 = F.normalize(x1, p=2, dim=-1)  # 归一化
        # monitor_resource(operation="GCN.forward - 前向传播完成", prefix="POST", mode="end")
        return x1

    def initialize(self):
        # monitor_resource(operation="GCN.initialize - 参数重置前", prefix="PRE", mode="start")
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        # monitor_resource(operation="GCN.initialize - 参数重置完成", prefix="POST", mode="end")

class GCN_LP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        # monitor_resource(operation="GCN_LP - 初始化前", prefix="PRE", mode="start")
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        # monitor_resource(operation="GCN_LP - 初始化完成（创建2层GCNConv）", prefix="POST", mode="end")

    def encode(self, x, edge_index):
        # monitor_resource(operation="GCN_LP.encode - 编码前", prefix="PRE", mode="start")
        # monitor_resource(operation="GCN_LP.encode - 第一层编码完成", prefix="MID", mode="start")
        x = self.conv1(x, edge_index).relu()  # 第一层编码
        # monitor_resource(operation="GCN_LP.encode - 第一层编码完成", prefix="MID", mode="end")
        x = self.conv2(x, edge_index)  # 第二层编码
        # monitor_resource(operation="GCN_LP.encode - 编码完成", prefix="POST", mode="end")
        return x

    def decode(self, z, edge_label_index):
        # monitor_resource(operation="GCN_LP.decode - 解码前", prefix="PRE", mode="start")
        # z所有节点的表示向量
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print(dst.size())   # (7284, 64)
        r = (src * dst).sum(dim=-1)  # 点积计算（高显存）
        # print(r.size())   (7284)
        # monitor_resource(operation="GCN_LP.decode - 解码完成", prefix="POST", mode="end")
        return r

    def forward(self, x, edge_index, edge_label_index):
        # monitor_resource(operation="GCN_LP.forward - 前向传播前", prefix="PRE", mode="start")
        z = self.encode(x, edge_index)
        result = self.decode(z, edge_label_index)
        # monitor_resource(operation="GCN_LP.forward - 前向传播完成", prefix="POST", mode="end")
        return result

class unnamed_model4(nn.Module):
    def __init__(self, args, input_size, hidden_size, output_size, num_classes):
        # monitor_resource(operation="unnamed_model4 - 初始化前", prefix="PRE", mode="start")
        super(unnamed_model4, self).__init__()
        self.args = args
        self.best_acc_pred_val1 = 0
        self.best_acc_pred_val2 = 0
        self.label_smoothing = None
        self.forget_rate = 0.9
        # monitor_resource(operation="unnamed_model4 - 初始化完成（基础参数设置）", prefix="POST", mode="end")

    def find_neighbour(self, adj, node_list):
        # monitor_resource(operation="find_neighbour - 查找邻居前", prefix="PRE", mode="start")
        # monitor_resource(operation="find_neighbour - 构建NetworkX图完成", prefix="MID1", mode="start")
        G = nx.from_numpy_array(adj.cpu().detach().numpy())  # GPU→CPU，显存减少
        # monitor_resource(operation="find_neighbour - 构建NetworkX图完成", prefix="MID1", mode="end")
        G.edges(data=True)
        neighbour = set()
        for node in node_list.cpu().detach().numpy().tolist():
            list_node = list(nx.neighbors(G, node))  # find 1_th neighbors
            for i in list_node:
                neighbour.add((node, i))
        # monitor_resource(operation="find_neighbour - 邻居查找完成", prefix="POST", mode="end")
        return neighbour

    def select_structure(self, y_1, t, forget_rate):
        # monitor_resource(operation="select_structure - 结构选择前", prefix="PRE", mode="start")
        confidence = y_1.max(1)[0]
        ind_2_sorted = np.argsort(confidence.cpu().data).cuda()  # CPU→GPU
        loss_2_sorted = confidence[ind_2_sorted]
        # mask=torch.where(loss_2_sorted >= 0.9).detach()

        # loss_2 = F.cross_entropy(y_2, t, reduce=False)
        # ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
        # loss_2_sorted = loss_2[ind_2_sorted]
        remember_rate2 = 1 - forget_rate
        num_remember2_1 = int(remember_rate2 * len(loss_2_sorted))
        # num_remember2_2 = mask.shape
        ind_2_update = ind_2_sorted[:num_remember2_1]
        # monitor_resource(operation="select_structure - 结构选择完成", prefix="POST", mode="end")
        return ind_2_update

    def select_feature(self, recon, x, forget_rate):
        # monitor_resource(operation="select_feature - 特征选择前", prefix="PRE", mode="start")
        loss = F.mse_loss(recon, x, reduce=False)  # MSE损失（高显存）
        loss = torch.sum(loss, dim=1)
        ind_sorted = np.argsort(loss.cpu().data).cuda()  # CPU→GPU
        loss_sorted = loss[ind_sorted]
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[num_remember:]
        # monitor_resource(operation="select_feature - 特征选择完成", prefix="POST", mode="end")
        return ind_update, loss

    def select_label(self, y_1, t, train_mask, forget_rate):
        # monitor_resource(operation="select_label - 标签选择前", prefix="PRE", mode="start")
        loss_1 = F.cross_entropy(y_1[train_mask], t[train_mask], reduce=False)  # 交叉熵损失（高显存）
        ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()  # CPU→GPU
        loss_1_sorted = loss_1[ind_1_sorted]

        # loss_2 = F.cross_entropy(y_2[train_mask], t[train_mask], reduce=False)
        # ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
        # loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember]
        # ind_2_update = ind_2_sorted[:num_remember]
        ind_1_update_final = train_mask[ind_1_update]
        # ind_2_update_final = train_mask[ind_2_update]
        # monitor_resource(operation="select_label - 标签选择完成", prefix="POST", mode="end")
        return ind_1_update_final

    def fs_c(self, model, features, edge_index, edge_weight, feature_recon, features_old, v, epoch):
        # monitor_resource(operation="fs_c - 特征与结构净化前", prefix="PRE", mode="start")
        # 学习weieght
        # knn利用feature的消融实验
        # representations, rec_loss_s, y = model(edge_index,self.args.features, self.features_sim, self.label_smoothing)
        # monitor_resource(operation="fs_c - 模型前向传播完成", prefix="MID1", mode="start")
        representations, rec_loss_s, y = model(edge_index, features, self.label_smoothing)  # 模型前向传播（高显存）
        # monitor_resource(operation="fs_c - 模型前向传播完成", prefix="MID1", mode="end")
        
        # # 融合结构
        # ones_matrix1 = torch.ones(self.pred_edge_index1.shape[1]) 
        # ones_matrix2 = torch.ones(self.pred_edge_index2.shape[1]) 
        # adj_knn1 = to_adj(self.pred_edge_index1, ones_matrix1, features.shape[0])  # 边→邻接矩阵（高显存）
        # adj_knn2 = to_adj(self.pred_edge_index2, ones_matrix2, features.shape[0])  # 边→邻接矩阵（高显存）
        # #显存增加较多
        # # monitor_resource(operation="fs_c - KNN邻接矩阵构建完成", prefix="MID2", mode="start")
        # # monitor_resource(operation="fs_c - KNN邻接矩阵构建完成", prefix="MID2", mode="end")
        # adj_fusion = self.fusion_adj(adj_knn1, adj_knn2, 1)  # 邻接矩阵融合
        # adj_fusion_cpu = adj_fusion.cpu().detach()  # 移到 CPU + 脱离计算图
        # # 步骤2：张量 → numpy 数组
        # adj_fusion_np = adj_fusion_cpu.numpy()
        # adj_fusion_coo = coo_matrix(adj_fusion_np)
        # pred_index, _ = to_edge_index(adj_fusion_coo)  # 邻接矩阵→边索引（高显存）
        pred_index= self.fusion_adj(self.pred_edge_index2, self.pred_edge_index1, 1)  # 边索引融合
        # monitor_resource(operation="fs_c - 结构融合完成", prefix="MID3", mode="start")
        # monitor_resource(operation="fs_c - 结构融合完成", prefix="MID3", mode="end")

        predictor_weights = model.get_estimated_weigths(pred_index.long(), representations)  # 估计边权重
        # predictor_weights_plus = model.get_estimated_weigths(self.pred_edge_index2, representations)
        edge_remain_idx = torch.where(predictor_weights != 0)[0].detach()
        # edge_add=torch.where(predictor_weights_plus >= 0.95)[0].detach()
        predictor_weights = predictor_weights[edge_remain_idx]
        pred_edge_index = pred_index[:, edge_remain_idx]
        # filepath1 = r'./structure_ori/{}.csv'.format(epoch)
        # df1=pd.DataFrame(edge_remain_idx.cpu().detach().numpy())
        # df1.to_csv(filepath1, index=False)
        
        tmp_edge = tu.remove_self_loops(edge_index=edge_index)[0]
        
        adj = dgl.graph((tmp_edge[0], tmp_edge[1]), num_nodes=features.shape[0])  # 构建DGL图
        adj = dgl.add_self_loop(adj)
        #显存增加较多
        # monitor_resource(operation="fs_c - DGL图构建完成", prefix="MID4", mode="start")
        # monitor_resource(operation="fs_c - DGL图构建完成", prefix="MID4", mode="end")

        # 融合feature
        features = torch.tensor(features, device='cuda:0')  # 确保特征在GPU上
        features_fusion = self.fusion_feature(self.args.features, features, 0)  # 特征融合
        #显存增加较多
        # monitor_resource(operation="fs_c - 特征融合完成", prefix="MID5", mode="start")
        # monitor_resource(operation="fs_c - 特征融合完成", prefix="MID5", mode="end")

        rec_loss_f, recon = feature_recon(adj.cpu(), predictor_weights.cpu(), features_fusion.cpu())  # 特征重构（CPU执行，显存减少）
        nodes, loss_f = self.select_feature(recon.cuda(), self.features, self.args.feature_noise)  # GPU→CPU→GPU，显存变化
        # monitor_resource(operation="fs_c - 特征重构与选择完成", prefix="MID6", mode="start")
        # monitor_resource(operation="fs_c - 特征重构与选择完成", prefix="MID6", mode="end")
        # filepath1 = r'./feature_truth/nodes.csv'
        # df1= pd.DataFrame(nodes.cpu().detach().numpy())
        # df1.to_csv(filepath1, index=False)
        # set1=set(nodes1)
        # set2=set(nodes2)
        # nodes=list(set2-set1)
        features_after = features.clone()  # 特征复制（显存增加）
        
        recon = torch.tensor(recon, device=self.args.device).clone()  # CPU→GPU，显存增加
        # aggregator = WeightedAggregation()
        # self.best_edge_idx1=self.edge_index.clone()
        # self.best_features1 = self.features1.clone()
        # self.best_pred_graph1 =self.edge_weight.clone()
        # feats = aggregator(self.best_features1, self.best_edge_idx1, self.best_pred_graph1)
        # monitor_resource(operation="fs_c - 特征聚合开始", prefix="MID7", mode="start")
        feats = scatter(features[tmp_edge[1]], tmp_edge[0], dim=0, dim_size=features.size(0), reduce='mean')
        # monitor_resource(operation="fs_c - 特征聚合完成", prefix="MID7", mode="end")
        # feats = aggregator(features, pred_edge_index, predictor_weights)
        #不更新features注释此行
        features_after[nodes] = feats[nodes].detach()  # 特征更新
        # recon_l=F.mse_loss(repr1,features)
        # consistency_loss=0
        # monitor_resource(operation="fs_c - 特征与结构净化完成", prefix="POST", mode="end")
        return features_after.cuda(), pred_edge_index.cuda(), predictor_weights.cuda(), rec_loss_s, rec_loss_f, representations, nodes, loss_f

    def generate_loss_matrix(self, losses):
        # monitor_resource(operation="generate_loss_matrix - 生成损失矩阵前", prefix="PRE", mode="start")
        """
        生成上三角loss和矩阵（仅保留i<j的组合，对角线置零）
        :param losses: 每个点的loss值列表
        :return: 上三角loss和矩阵
        """
        losses = np.array(losses.cpu().detach())  # GPU→CPU，显存减少
        # 生成i<j的上三角矩阵，其他位置自动置零
        mat = np.triu(losses[:, np.newaxis] + losses[np.newaxis, :], k=1)
        # monitor_resource(operation="generate_loss_matrix - 损失矩阵生成完成", prefix="POST", mode="end")
        return mat

    def get_top_coordinates(self, mat, percent=0.3):
        # monitor_resource(operation="get_top_coordinates - 获取Top坐标前", prefix="PRE", mode="start")
        """
        获取非零元素中最大的percent比例的坐标
        :param mat: 上三角loss和矩阵
        :param percent: 比例（0-1）
        :return: 降序排列的坐标列表
        """
        # 获取所有非零元素的坐标
        coords = np.argwhere(mat > 0)
        if len(coords) == 0:
            # monitor_resource(operation="get_top_coordinates - 无有效坐标", prefix="POST", mode="end")
            return []
        values = mat[coords[:, 0], coords[:, 1]]

        # 计算选取数量
        k = int(np.ceil(percent * len(coords)))
        if k == 0:
            # monitor_resource(operation="get_top_coordinates - 选取数量为0", prefix="POST", mode="end")
            return []

        # 高效获取topk索引并排序
        indices = np.argpartition(-values, k - 1)[:k]
        sorted_indices = indices[np.argsort(-values[indices])]

        result = [tuple(coords[i]) for i in sorted_indices]
        # monitor_resource(operation="get_top_coordinates - Top坐标获取完成", prefix="POST", mode="end")
        return result

    def remove_edges_from_adjacency(self, adj_matrix, edges_to_remove):
        # monitor_resource(operation="remove_edges_from_adjacency - 邻接矩阵删边前", prefix="PRE", mode="start")
        """
        从邻接矩阵中删除指定的边
        :param adj_matrix: 图的邻接矩阵
        :param edges_to_remove: 要删除的边的坐标列表，例如[(i1, j1), (i2, j2), ...]
        :return: 更新后的邻接矩阵
        """
        # 复制邻接矩阵以避免修改原始数据
        updated_adj_matrix = adj_matrix.clone()  # 矩阵复制（显存增加）

        # 遍历要删除的边，将对应位置的值设为0
        for i, j in edges_to_remove:
            updated_adj_matrix[i, j] = 0
            updated_adj_matrix[j, i] = 0  # 如果是无向图，需要同时删除对称位置

        # monitor_resource(operation="remove_edges_from_adjacency - 邻接矩阵删边完成", prefix="POST", mode="end")
        return updated_adj_matrix

    def fit(self):
        # monitor_resource(operation="unnamed_model4.fit - 训练开始前", prefix="PRE", mode="start")
        self.args.num_heads = 4
        self.args.num_out_heads = 1
        self.args.num_hidden = 256
        self.args.num_layers = 2
        self.args.residual = False
        self.args.attn_drop = .1
        self.args.in_drop = .2
        self.args.norm = None
        self.args.negative_slope = 0.2
        self.args.encoder = "gat"
        self.args.decoder = "gat"
        self.args.mask_rate = 0.5
        self.args.drop_edge_rate = 0  # 随机mask边
        self.args.replace_rate = 0.0
        self.args.activation = "prelu"
        self.args.loss_fn = "sce"
        self.args.alpha_l = 2
        self.args.concat_hidden = False
        self.args.num_features = self.args.features.shape[1]
        args = self.args

        self.edge_index = args.edge_index
        self.edge_weight = args.edge_weight
        self.best_edge_idx1 = self.edge_index.clone()  # 边索引复制（显存增加）
        self.best_features1 = self.args.features1.clone()  # 特征复制（显存增加）
        self.best_pred_graph1 = self.edge_weight.clone()  # 边权重复制（显存增加）
        # monitor_resource(operation="unnamed_model4.fit - 基础参数与初始数据复制完成", prefix="MID1", mode="start")
        # monitor_resource(operation="unnamed_model4.fit - 基础参数与初始数据复制完成", prefix="MID1", mode="end")

        self.features = args.features.clone()  # 特征复制（显存增加）
        self.labels = args.labels.clone()  # 标签复制（显存增加）
        self.idx_train = args.train_index
        self.idx_val = args.val_index
        self.idx_test = args.test_index
        self.device = args.device
        self.labels_noise = args.labels_noise
        # monitor_resource(operation="unnamed_model4.fit - 数据初始化完成", prefix="MID2", mode="start")
        # monitor_resource(operation="unnamed_model4.fit - 数据初始化完成", prefix="MID2", mode="end")

        # 初始化子模型（高显存消耗）
        self.q_phi1 = q_phi1(self.features, self.features.shape[1], args, device=self.device).to(self.device)
        # monitor_resource(operation="unnamed_model4.fit - q_phi1初始化完成", prefix="MID3", mode="start")
        # monitor_resource(operation="unnamed_model4.fit - q_phi1初始化完成", prefix="MID3", mode="end")
        self.q_phi3 = GCN(nfeat=self.features.shape[1],
                          nhid=self.args.hidden_size,
                          nclass=self.labels.max().item() + 1,
                          dropout=self.args.dropout, device=self.device).to(self.device)
        # monitor_resource(operation="unnamed_model4.fit - q_phi3（GCN）初始化完成", prefix="MID4", mode="start")
        # monitor_resource(operation="unnamed_model4.fit - q_phi3（GCN）初始化完成", prefix="MID4", mode="end")
        # self.q_phi2 = q_phi1(self.features, self.features.shape[1], args, device=self.device).to(self.device)

        # self.q_phi4 = GCN(nfea=self.features.shape[1],
        #                   nhid=self.args.hidden_size,
        #                   nclass=self.labels.max().item() + 1,
        #                   dropout=self.args.dropout, device=self.device).to(self.device)
        self.feature_recon1 = build_model(self.args)  # 构建特征重构模型（高显存）
        # monitor_resource(operation="unnamed_model4.fit - feature_recon1初始化完成", prefix="MID5", mode="start")
        # monitor_resource(operation="unnamed_model4.fit - feature_recon1初始化完成", prefix="MID5", mode="end")
        # self.feature_recon2 = build_model(self.args)
        # self.fusion_adj =DimWiseFusion(self.args.adj.shape).to(self.device)  # 邻接矩阵融合模型（显存增加）
        # self.fusion_feature =DimWiseFusion(self.features.shape).to(self.device)  # 特征融合模型（显存增加）

        self.fusion_adj = DimWiseFusion().to(self.device)  # 邻接矩阵融合模型（显存增加）
        self.fusion_feature = DimWiseFusion().to(self.device)  # 特征融合模型（显存增加）
        # monitor_resource(operation="unnamed_model4.fit - 融合模型初始化完成", prefix="MID6", mode="start")
        # monitor_resource(operation="unnamed_model4.fit - 融合模型初始化完成", prefix="MID6", mode="end")
       
        # self.features_sim = F.normalize(self.features, dim=1, p=2).mm(
        #     F.normalize(self.features, dim=1, p=2).T).fill_diagonal_(0.0)  # 特征相似度矩阵（高显存，矩阵乘法）
        #显存增加较多
        # monitor_resource(operation="unnamed_model4.fit - 特征相似度矩阵计算完成", prefix="MID7", mode="start")
        # monitor_resource(operation="unnamed_model4.fit - 特征相似度矩阵计算完成", prefix="MID7", mode="end")
        
        # Prior of Z_A
        if self.args.K > 0:
            if self.args.hop == 0:
                feats = self.features.clone()  # 特征复制
            elif self.args.hop > 0:
                tmp_edge = tu.add_self_loops(edge_index=self.edge_index, num_nodes=self.features.shape[0])[0]
                feats = scatter(self.features[tmp_edge[1]], tmp_edge[0], dim=0, dim_size=self.features.size(0),
                                reduce='mean')  # 特征聚合（高显存）
            # sim = F.normalize(feats, dim=1, p=2).mm(F.normalize(feats, dim=1, p=2).T).fill_diagonal_(0.0)  # 相似度矩阵（高显存）
            # dst = sim.topk(self.args.K, 1)[1].to(self.device)
            # src = torch.arange(sim.size(0)).unsqueeze(1).expand_as(dst).to(self.device)
            # self.knn_edge = torch.stack([src.reshape(-1), dst.reshape(-1)])
            # self.pred_edge_index1 = tu.to_undirected(self.knn_edge).to(self.device)  # KNN边构建（显存增加）
            # self.pred_edge_index =edge_index.clone()
            self.pred_edge_index1=get_top_k_similar_edge_index(feats, K=self.args.K)  # KNN边构建（显存增加）
        else:
            self.pred_edge_index1 = self.edge_index.clone()  # 边索引复制
        self.pred_edge_index1 = tu.coalesce(
            torch.cat([self.edge_index.to(self.device), self.pred_edge_index1.to(self.device)], dim=1))  # 边合并去重（显存增加）
        self.pred_edge_index2 = self.pred_edge_index1.clone()  # 边索引复制（显存增加）
        #显存增加较多
        # monitor_resource(operation="unnamed_model4.fit - KNN边构建完成", prefix="MID8", mode="start")
        # monitor_resource(operation="unnamed_model4.fit - KNN边构建完成", prefix="MID8", mode="end")
        
        # 初始化优化器（参数较多，显存增加）
        self.optimizer1 = optim.Adam(
            list(self.q_phi1.parameters()) + list(self.q_phi3.parameters()) + 
            list(self.feature_recon1.parameters()) + list(self.fusion_adj.parameters()) + 
            list(self.fusion_feature.parameters()), 
            lr=args.lr, weight_decay=args.wd
        )
        # monitor_resource(operation="unnamed_model4.fit - 优化器初始化完成", prefix="MID9", mode="start")
        # monitor_resource(operation="unnamed_model4.fit - 优化器初始化完成", prefix="MID9", mode="end")
        # self.optimizer2 = optim.Adam(list(self.q_phi2.parameters()) + list(self.q_phi4.parameters())+list(self.feature_recon2.parameters()), lr=args.lr, weight_decay=args.wd)
        
        cnt_wait = 0
        acc_list = []
        str_loss_list = []
        fea_loss_list = []
        training_time_list = []
        # monitor_resource(operation="unnamed_model4.fit - 训练循环准备完成，开始迭代", prefix="MID10", mode="start")
        # monitor_resource(operation="unnamed_model4.fit - 训练循环准备完成，开始迭代", prefix="MID10", mode="end")
        
        for epoch in range(args.epochs):
            epoch_key = f"EPOCH_{epoch+1}"
            # monitor_resource(operation=f"unnamed_model4.fit - {epoch_key} 开始", prefix=epoch_key+"_PRE", mode="start")
            val_acc, t = self.train(epoch, self.features, self.edge_index, self.edge_weight, self.idx_train, self.idx_val, self.idx_test)
            acc_list.append(val_acc.item())
            # str_loss_list.append(str_loss.item())
            # fea_loss_list.append(fea_loss.item())
            training_time_list.append(t)
            # monitor_resource(operation=f"unnamed_model4.fit - {epoch_key} 训练完成", prefix=epoch_key+"_POST", mode="end")
            
            if val_acc >= self.best_acc_pred_val1:
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                print(f'Early stopping at {epoch_key}!')
                break
        
        print("training_time:{:.4f}".format(np.mean(np.array(training_time_list))),"{:.4f}".format(np.std(np.array(training_time_list))))
        # monitor_resource(operation="unnamed_model4.fit - 训练循环结束", prefix="MID11", mode="start")
        # monitor_resource(operation="unnamed_model4.fit - 训练循环结束", prefix="MID11", mode="end")

        print("Optimization Finished!")
        print("picking the best model according to validation performance")

        self.q_phi3.load_state_dict(self.predictor_model_weigths1)  # 加载最优权重（显存变化）
        # monitor_resource(operation="unnamed_model4.fit - 加载最优模型权重完成", prefix="MID12", mode="start")
        # monitor_resource(operation="unnamed_model4.fit - 加载最优模型权重完成", prefix="MID12", mode="end")
        # self.q_phi4.load_state_dict(self.predictor_model_weigths2)
        # if not os.path.exists('./teacher_models/{}/g_{}_l_{}_f_{}'.format(args.dataset,args.graph_noise,args.label_noise,args.feature_noise)):
        #                 os.mkdir('./teacher_models/{}/g_{}_l_{}_f_{}'.format(args.dataset,args.graph_noise,args.label_noise,args.feature_noise))
        
        save_path = f"./saved_model_weights/{self.args.dataset}.pth"

        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存模型权重
        torch.save(self.predictor_model_weigths1, save_path)
        print(f"model weights saved: {save_path}")
        # monitor_resource(operation="unnamed_model4.fit - 训练完成，模型权重保存", prefix="POST", mode="end")
        
        # 输出时间统计汇总
        # self.print_time_summary()

    def train(self, epoch, features, edge_index, edge_weight, train_mask, val_mask, test_mask):
        epoch_key = f"TRAIN_EPOCH_{epoch+1}"
        # monitor_resource(operation=f"unnamed_model4.train - {epoch_key} 训练前", prefix=epoch_key+"_PRE", mode="start")
        self.q_phi1.train()
        # self.q_phi2.train()
        self.q_phi3.train()
        # self.q_phi4.train()
        self.feature_recon1.train()
        # self.feature_recon2.train()
      
        
        t = time.time()
        args = self.args

        flag = 1
        # monitor_resource(operation=f"unnamed_model4.train - 优化器梯度清零", prefix=epoch_key+"_MID1", mode="start")
        self.optimizer1.zero_grad()  # 优化器梯度清零（显存无明显变化）
        # monitor_resource(operation=f"unnamed_model4.train - 优化器梯度清零", prefix=epoch_key+"_MID1", mode="end")
        # self.optimizer2.zero_grad()
       
        # 进行feature和structure的净化（高显存消耗）
        args.features1, self.pred_edge_index1, predictor_weights1, rec_loss_s1, rec_loss_f1, representations1, nodes1, loss_f1 = self.fs_c(
            self.q_phi1, args.features1, edge_index, edge_weight, self.feature_recon1, args.features_old_ind1, 1, epoch)
        # monitor_resource(operation=f"unnamed_model4.train - 特征与结构净化完成", prefix=epoch_key+"_MID2", mode="start")
        # monitor_resource(operation=f"unnamed_model4.train - 特征与结构净化完成", prefix=epoch_key+"_MID2", mode="end")
        # args.features2, pred_edge_index2, predictor_weights2, rec_loss_s2, rec_loss_f2, representations2,nodes2,loss_f2 = self.fs_c(
        #     self.G,self.q_phi2, args.features2, edge_index, self.feature_recon2,args.features_old_ind2,2)
        
        pred_edge_index1 = self.pred_edge_index1
        features1 = args.features1
        
        # 使用coteaching选取高置信度样本
        output_z_y1 = self.q_phi3(features1.data, pred_edge_index1.data, predictor_weights1.data)  # GCN前向传播（高显存）
        # monitor_resource(operation=f"unnamed_model4.train - q_phi3前向传播完成", prefix=epoch_key+"_MID3", mode="start")
        # monitor_resource(operation=f"unnamed_model4.train - q_phi3前向传播完成", prefix=epoch_key+"_MID3", mode="end")
        # output_z_y2 = self.q_phi4(args.features2.data, pred_edge_index2.data, predictor_weights2.data)
        
        if self.args.teacher_student == 'student':
            label_index1 = self.select_label(self.args.soft_label1, self.args.targets_oneHot1, args.train_index_clean, args.label_noise)
            label_index2 = self.select_label(self.args.soft_label2, self.args.targets_oneHot1, args.train_index_clean, args.label_noise)
            label_index3 = self.select_label(self.args.soft_label3, self.args.targets_oneHot1, args.train_index_clean, args.label_noise)
            label_index4 = self.select_label(self.args.soft_label4, self.args.targets_oneHot1, args.train_index_clean, args.label_noise)
            intersection = label_index1
            intersection = tensor_intersection(intersection, label_index2)
            intersection = tensor_intersection(intersection, label_index3)
            intersection = tensor_intersection(intersection, label_index4)
            # monitor_resource(operation=f"unnamed_model4.train - 学生模式标签交集计算完成", prefix=epoch_key+"_MID4", mode="start")
            # monitor_resource(operation=f"unnamed_model4.train - 学生模式标签交集计算完成", prefix=epoch_key+"_MID4", mode="end")
            
            log_p = F.log_softmax(output_z_y1, dim=1)
            q1 = F.softmax(self.args.soft_label1, dim=1)
            q2 = F.softmax(self.args.soft_label2, dim=1)
            q3 = F.softmax(self.args.soft_label3, dim=1)
            q4 = F.softmax(self.args.soft_label4, dim=1)
            kld1 = F.kl_div(log_p, q1)
            kld2 = F.kl_div(log_p, q2)
            kld3 = F.kl_div(log_p, q3)
            kld4 = F.kl_div(log_p, q4)
            # monitor_resource(operation=f"unnamed_model4.train - KL散度计算完成", prefix=epoch_key+"_MID5", mode="start")
            # monitor_resource(operation=f"unnamed_model4.train - KL散度计算完成", prefix=epoch_key+"_MID5", mode="end")
        
        # 计算损失（高显存消耗）
        # monitor_resource(operation=f"unnamed_model4.train - 损失计算完成", prefix=epoch_key+"_MID6", mode="start")
        if self.args.teacher_student == 'student':
            loss1 = F.cross_entropy(output_z_y1[intersection], args.targets_oneHot1[intersection])
            loss_1 = F.cross_entropy(output_z_y1[train_mask], args.targets_oneHot1[train_mask], reduce=False)
        else:
            # loss1 =cosine_error_loss(output_z_y1[train_mask], args.targets_oneHot1[train_mask])
            loss1 = F.cross_entropy(output_z_y1[train_mask], args.targets_oneHot1[train_mask])
            loss_1 = F.cross_entropy(output_z_y1[train_mask], args.targets_oneHot1[train_mask], reduce=False)
        # monitor_resource(operation=f"unnamed_model4.train - 损失计算完成", prefix=epoch_key+"_MID6", mode="end")
        
        # 构建总损失
        # monitor_resource(operation=f"unnamed_model4.train - 总损失构建完成", prefix=epoch_key+"_MID7", mode="start")
        if self.args.teacher_student == 'student':
            if self.args.ntea == 4:
                loss_all1 = loss1 + 0.03 * rec_loss_s1 + rec_loss_f1 + (kld1 + kld2 + kld3 + kld4) * self.args.lambda_kld
            elif self.args.ntea == 3:
                loss_all1 = loss1 + 0.03 * rec_loss_s1 + rec_loss_f1 + (kld1 + kld2 + kld3) * self.args.lambda_kld
            elif self.args.ntea == 2:
                loss_all1 = loss1 + 0.03 * rec_loss_s1 + rec_loss_f1 + (kld1 + kld2) * self.args.lambda_kld
            elif self.args.ntea == 1:
                loss_all1 = loss1 + 0.03 * rec_loss_s1 + rec_loss_f1 + (kld1) * self.args.lambda_kld
        else:
            loss_all1 = loss1 + 0.03 * rec_loss_s1 + rec_loss_f1 
        # monitor_resource(operation=f"unnamed_model4.train - 总损失构建完成", prefix=epoch_key+"_MID7", mode="end")
        
        # 反向传播（高显存消耗，计算梯度）
        # monitor_resource(operation=f"unnamed_model4.train - 反向传播完成（梯度计算）", prefix=epoch_key+"_MID8", mode="start")
        loss_all1.backward()
        # monitor_resource(operation=f"unnamed_model4.train - 反向传播完成（梯度计算）", prefix=epoch_key+"_MID8", mode="end")
        
        # 参数更新（显存无明显变化）
        # monitor_resource(operation=f"unnamed_model4.train - 优化器参数更新完成", prefix=epoch_key+"_MID9", mode="start")
        self.optimizer1.step()
        # monitor_resource(operation=f"unnamed_model4.train - 优化器参数更新完成", prefix=epoch_key+"_MID9", mode="end")
        
        # 验证阶段（eval模式）
        self.q_phi3.eval()
        with torch.no_grad():  # 禁用梯度计算，显存减少
            output0 = self.q_phi3(features1, pred_edge_index1, predictor_weights1)  # 验证前向传播
            # monitor_resource(operation=f"unnamed_model4.train - 验证前向传播完成", prefix=epoch_key+"_MID10", mode="start")
            # monitor_resource(operation=f"unnamed_model4.train - 验证前向传播完成", prefix=epoch_key+"_MID10", mode="end")
            acc_pred_train0 = accuracy(output0[train_mask], args.targets_c_oneHot[train_mask])
            acc_pred_val0 = accuracy(output0[val_mask], args.targets_c_oneHot[val_mask])
            acc_pred_test0 = accuracy(output0[test_mask], args.targets_c_oneHot[test_mask])
            # monitor_resource(operation=f"unnamed_model4.train - 准确率计算完成", prefix=epoch_key+"_MID11", mode="start")
            # monitor_resource(operation=f"unnamed_model4.train - 准确率计算完成", prefix=epoch_key+"_MID11", mode="end")

            # adj_pred = to_adj(pred_edge_index1, predictor_weights1, features1.shape[0])  # 边→邻接矩阵（高显存）
            # str_loss = F.mse_loss(self.args.clean_adj, adj_pred)  # 结构损失
            # fea_loss = F.mse_loss(self.args.clean_features, features1)  # 特征损失
            #显存增加较多
            # monitor_resource(operation=f"unnamed_model4.train - 结构/特征损失计算完成", prefix=epoch_key+"_MID12", mode="start")
            # monitor_resource(operation=f"unnamed_model4.train - 结构/特征损失计算完成", prefix=epoch_key+"_MID12", mode="end")

        pre_value_max, pre_index_max = output_z_y1.max(1)
        t_time = time.time() - t
        
        # 保存最优模型
        if acc_pred_val0 >= self.best_acc_pred_val1:
            print(1)
            self.best_acc_pred_val1 = acc_pred_val0
            self.best_features1 = features1.detach()  # 分离特征（显存变化）
            self.best_pred_graph1 = predictor_weights1.detach()  # 分离边权重（显存变化）
            self.best_edge_idx1 = pred_edge_index1.detach()  # 分离边索引（显存变化）
            self.predictor_model_weigths1 = deepcopy(self.q_phi3.state_dict())  # 深拷贝权重（显存增加）
            better_inex = 1
            # monitor_resource(operation=f"unnamed_model4.train - 更新最优模型参数", prefix=epoch_key+"_MID13", mode="start")
            # monitor_resource(operation=f"unnamed_model4.train - 更新最优模型参数", prefix=epoch_key+"_MID13", mode="end")

        print('Epoch: {:04d}'.format(epoch + 1),
              'acc_train: {:.4f}'.format(acc_pred_train0.item()),
              'acc_val: {:.4f}'.format(acc_pred_val0.item()),
              'acc_test: {:.4f}'.format(acc_pred_test0.item()),
              'loss:{:.4f}'.format(loss_all1.item()),
              'time: {:.4f}s'.format(t_time)),

        if epoch <= 30:
            with torch.no_grad():
                tmp_edge = edge_index[:, edge_index[0] < edge_index[1]]
                if epoch == 0:
                    self.phat = torch.ones_like(tmp_edge[0]).float()
                phat = F.relu(F.cosine_similarity(representations1[tmp_edge[0]], representations1[tmp_edge[1]], dim=1))
                self.phat = self.args.decay * self.phat + (1 - self.args.decay) * phat
                min_value, max_value = self.phat.min(), self.phat.max()
                normalized_tensor = (self.phat - min_value) / (max_value - min_value)
                self.label_smoothing = 0.1 * normalized_tensor + 0.9
                # monitor_resource(operation=f"unnamed_model4.train - 标签平滑更新完成", prefix=epoch_key+"_MID14", mode="start")
                # monitor_resource(operation=f"unnamed_model4.train - 标签平滑更新完成", prefix=epoch_key+"_MID14", mode="end")

        # monitor_resource(operation=f"unnamed_model4.train - {epoch_key} 训练结束", prefix=epoch_key+"_POST", mode="start")
        # monitor_resource(operation=f"unnamed_model4.train - {epoch_key} 训练结束", prefix=epoch_key+"_POST", mode="end")
        return acc_pred_val0, t_time

    def test(self, idx_test, seed, noise_type):
        # monitor_resource(operation="unnamed_model4.test - 测试前", prefix="TEST_PRE", mode="start")
        self.q_phi3.eval()
        estimated_weights1 = self.best_pred_graph1
        pred_edge_index1 = self.best_edge_idx1
        features1 = self.best_features1
        t1 = time.time()
        with torch.no_grad():  # 禁用梯度计算，显存减少
            output0 = self.q_phi3(features1, pred_edge_index1, estimated_weights1)  # 测试前向传播（高显存）
            # monitor_resource(operation="unnamed_model4.test - 测试前向传播完成", prefix="TEST_MID1", mode="start")
            # monitor_resource(operation="unnamed_model4.test - 测试前向传播完成", prefix="TEST_MID1", mode="end")
        t2 = time.time()
        
        # 保存教师模型软标签
        if self.args.teacher_record and self.args.teacher_student == 'teacher':
            save_dir = f'./teacher_models/{noise_type}/{self.args.dataset}/g_{self.args.graph_noise}_l_{self.args.label_noise}_f_{self.args.feature_noise}'
            os.makedirs(save_dir, exist_ok=True)
            part_suffix = f"_part{self.args.part_id}" if getattr(self.args, 'part_id', None) is not None else ""
            with open(f'{save_dir}/soft_label{seed}{part_suffix}.pkl', 'wb') as f:
                pickle.dump(output0, f)
            # monitor_resource(operation="unnamed_model4.test - 教师模型软标签保存完成", prefix="TEST_MID2", mode="start")
            # monitor_resource(operation="unnamed_model4.test - 教师模型软标签保存完成", prefix="TEST_MID2", mode="end")
        
        # 计算测试指标
        acc_pred_test0 = accuracy(output0[idx_test], self.args.targets_c_oneHot[idx_test]).item()
        pre, rec, f1 = calculate_classification_metrics(output0[idx_test], self.args.targets_c_oneHot[idx_test])
        print("Test Accuray: #1 = %f" % (acc_pred_test0))
        # monitor_resource(operation="unnamed_model4.test - 测试完成", prefix="TEST_POST", mode="end")
        return acc_pred_test0, output0.detach(), pre, rec, f1, t2 - t1

    # def print_time_summary(self):
    #     """输出所有操作的时间统计汇总"""
    #     print("\n" + "="*80)
    #     print("                          运行时间统计汇总")
    #     print("="*80)
    #     # 按耗时排序
    #     sorted_times = sorted(time_stats.items(), key=lambda x: x[1], reverse=True)
    #     total_time = sum([t for _, t in sorted_times])
    #     print(f"总运行时间: {total_time:.4f}s ({total_time/60:.2f}分钟)")
    #     print("\n各操作耗时详情（按耗时降序排列）:")
    #     print("-"*80)
    #     print(f"{'操作标识':<50} {'耗时(s)':<15} {'占比(%)':<10}")
    #     print("-"*80)
    #     for op, elapsed in sorted_times:
    #         ratio = (elapsed / total_time) * 100
    #         print(f"{op:<50} {elapsed:.4f}        {ratio:.2f}")
    #     print("="*80 + "\n")

class q_phi1(nn.Module):
    def __init__(self, features, nfea, args, device='cuda'):
        # monitor_resource(operation="q_phi1 - 初始化前", prefix="PRE", mode="start")
        super(q_phi1, self).__init__()
        self.estimator = GCN(nfea, args.hidden_size, args.hidden_size, dropout=0.0, device=device)  # GCN初始化（显存增加）
        self.device = device
        self.args = args
        self.y_fc = nn.Linear(args.hidden_size, args.num_classes)  # 全连接层（显存增加）
        self.feature_ori = features.clone()  # 原始特征复制（显存增加）
        # monitor_resource(operation="q_phi1 - 初始化完成（GCN+全连接层）", prefix="POST", mode="end")
        

    def forward(self, edge_index, features, label_smoothing):
        # monitor_resource(operation="q_phi1.forward - 前向传播前", prefix="PRE", mode="start")
        representations = self.estimator(features, edge_index, \
                                         torch.ones([edge_index.shape[1]]).to(self.device).float())  # GCN前向传播（高显存）
        # monitor_resource(operation="q_phi1.forward - GCN特征提取完成", prefix="MID1", mode="start")
        # monitor_resource(operation="q_phi1.forward - GCN特征提取完成", prefix="MID1", mode="end")
        representations = F.normalize(representations, dim=-1)  # 归一化
        # print(representations.shape)
        # monitor_resource(operation="q_phi1.forward - 全连接层预测完成", prefix="MID2", mode="start")
        y = self.y_fc(representations)  # 全连接层预测（显存变化）
        # monitor_resource(operation="q_phi1.forward - 全连接层预测完成", prefix="MID2", mode="end")
        rec_loss, pos_w = self.reconstruct_loss(edge_index, representations, label_smoothing)  # 重构损失（高显存）
        # monitor_resource(operation="q_phi1.forward - 前向传播完成", prefix="POST", mode="end")
        return representations, rec_loss, y

    def get_estimated_weigths(self, edge_index, representations):
        # monitor_resource(operation="get_estimated_weigths - 估计边权重前", prefix="PRE", mode="start")
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0, x1), dim=1)  # 点积计算边权重（高显存）
        estimated_weights = F.relu(output)
        # estimated_weights = torch.cat([x0, x1], dim=1)
        if estimated_weights.shape[0] != 0:
            estimated_weights = estimated_weights.masked_fill(estimated_weights < self.args.tau, 0.0)  # 阈值过滤
        # monitor_resource(operation="get_estimated_weigths - 边权重估计完成", prefix="POST", mode="end")
        return estimated_weights

    def reconstruct_loss(self, edge_index, representations, label_smoothing):
        # monitor_resource(operation="reconstruct_loss - 计算重构损失前", prefix="PRE", mode="start")
        num_nodes = representations.shape[0]
        randn = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=edge_index.size(1))  # 负采样（显存增加）
        randn = randn[:, randn[0] < randn[1]]
        # monitor_resource(operation="reconstruct_loss - 负采样完成", prefix="MID1", mode="start")
        # monitor_resource(operation="reconstruct_loss - 负采样完成", prefix="MID1", mode="end")

        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg_w = F.relu(torch.sum(torch.mul(neg0, neg1), dim=1))  # 负样本权重（高显存）
        # monitor_resource(operation="reconstruct_loss - 负样本权重计算完成", prefix="MID2", mode="start")
        # monitor_resource(operation="reconstruct_loss - 负样本权重计算完成", prefix="MID2", mode="end")

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos_w = F.relu(torch.sum(torch.mul(pos0, pos1), dim=1))  # 正样本权重（高显存）
        # monitor_resource(operation="reconstruct_loss - 正样本权重计算完成", prefix="MID3", mode="start")
        # monitor_resource(operation="reconstruct_loss - 正样本权重计算完成", prefix="MID3", mode="end")

        # pos_fsim = sim[edge_index[0], edge_index[1]]
        u = edge_index[0]  # 源节点列表
        v = edge_index[1]  # 目标节点列表
        
        # 提取 u 和 v 对应的特征向量（形状均为 [E, D]）
        feat_u = self.feature_ori[u]  # 每个 u 的特征：[E, D]
        feat_v = self.feature_ori[v]  # 每个 v 的特征：[E, D]
        
        # 计算每个点对的内积（相似度）：按行点积，结果形状 [E]
        pos_fsim = torch.sum(feat_u * feat_v, dim=1)  # 等价于 (feat_u * feat_v).sum(dim=1)

        pos = pos_fsim * self.args.theta + pos_w * (1 - self.args.theta)  # 融合权重
        # monitor_resource(operation="reconstruct_loss - 正负样本权重融合完成", prefix="MID4", mode="start")
        # monitor_resource(operation="reconstruct_loss - 正负样本权重融合完成", prefix="MID4", mode="end")

        # 计算重构损失（高显存）
        if label_smoothing is None:
            rec_loss = (F.mse_loss(neg_w, torch.zeros_like(neg_w), reduction='sum') \
                        + F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) \
                        * num_nodes / (randn.shape[1] + edge_index.shape[1])
        else:
            rec_loss = (F.mse_loss(neg_w, torch.zeros_like(neg_w), reduction='sum') \
                        + F.mse_loss(pos, label_smoothing, reduction='sum')) \
                        * num_nodes / (randn.shape[1] + edge_index.shape[1])

        # monitor_resource(operation="reconstruct_loss - 重构损失计算完成", prefix="POST", mode="end")
        return rec_loss, pos_w