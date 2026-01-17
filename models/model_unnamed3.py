import filecmp
import dgl.function as fn
import torch.optim as optim
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from models.encoder_decoder import *
from scipy.sparse import coo_matrix
from models.Semi_RNCGLN import *
from torch_geometric.utils import to_undirected, dense_to_sparse, to_dense_adj, is_undirected, remove_self_loops, \
    degree, to_scipy_sparse_matrix, from_scipy_sparse_matrix
import torch_geometric.utils as tu
from torch.autograd import Variable
# from torch_sparsemax import Sparsemax
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
def tensor_intersection(tensor1, tensor2):
    mask = torch.isin(tensor1, tensor2)
    intersection = tensor1[mask]
    return torch.unique(intersection)
def calculate_classification_metrics(logits, targets_onehot, num_classes=None, average='macro'):
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
    preds_indices = logits.argmax(dim=1)
    
    # 将one-hot标签转换为类别索引
    targets_indices = targets_onehot.argmax(dim=1)
    
    # 自动推断类别数
    if num_classes is None:
        num_classes = max(logits.shape[1], targets_indices.max().item() + 1)
    
    # 计算混淆矩阵元素
    tp = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    fp = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    fn = torch.zeros(num_classes, dtype=torch.float32, device=logits.device)
    
    for c in range(num_classes):
        # 当前类别的预测和真实标签
        preds_c = (preds_indices == c)
        targets_c = (targets_indices == c)
        
        tp[c] = (preds_c & targets_c).sum()
        fp[c] = (preds_c & (~targets_c)).sum()
        fn[c] = ((~preds_c) & targets_c).sum()
    
    # 计算各类别的精确率、召回率和F1分数
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
    
    return precision, recall, f1_score


class DimWiseFusion(nn.Module):
    def __init__(self, dim):
        super(DimWiseFusion, self).__init__()
        # 初始化与特征维度相同的可学习权重w
        self.w = nn.Parameter(torch.randn(dim))
        # 使用sigmoid确保每个维度的w在[0,1]范围内
        
    def forward(self, A0, An,ver):
        w = torch.sigmoid(self.w)  # 每个维度的权重独立控制在[0,1]
        if ver==1:
            w=(w > 0.5).float()
        return w * A0 + (1 - w) * An  # 点乘融合
def adjacency_ce_loss(adj1, adj2):
    """
    计算两个邻接矩阵之间的交叉熵损失
    
    参数:
        adj1 (torch.Tensor): 第一个邻接矩阵(概率)
        adj2 (torch.Tensor): 第二个邻接矩阵(0/1)
        
    返回:
        torch.Tensor: 交叉熵损失值
    """
    ce_loss = nn.BCELoss()
    return ce_loss(adj1, adj2)

def remove_duplicate_edges(edge_index, edge_weights):
    # 转置 edge_index 并拼接权重，便于去重
    edges_with_weights = torch.cat([
        edge_index.t(),
        edge_weights.unsqueeze(1)
    ], dim=1)  # shape: [num_edges, 3] (u, v, weight)

    # 对 (u, v) 去重，保留第一个出现的边
    unique_edges, indices = torch.unique(
        edges_with_weights[:, :2],  # 仅针对 (u, v) 去重
        dim=0,
        return_inverse=True,
        return_counts=False
    )

    # 获取去重后的权重（取每组重复边的第一个权重）
    unique_weights = edge_weights[indices.unique()]

    # 转置回 [2, num_unique_edges] 格式
    unique_edge_index = unique_edges.t().to(torch.long)

    return unique_edge_index, unique_weights

class WeightedAggregation(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # 初始聚合方式设为求和
    
    def forward(self, x, edge_index, edge_weight):
        # 移除自环边
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        
        # 计算归一化权重（可选）
        row, col = edge_index
        deg = torch.zeros(x.size(0), device=x.device)
        deg.scatter_add_(0, col, edge_weight)  # 按目标节点累加权重
        deg_inv = 1.0 / deg
        deg_inv[deg == 0] = 0  # 处理无邻居节点
        norm_weight = edge_weight * deg_inv[col]  # 归一化权重
        
        # 消息传递（加权平均）
        return self.propagate(edge_index, x=x, norm_weight=norm_weight)
    
    def message(self, x_j, norm_weight):
        return x_j * norm_weight.view(-1, 1)  # 消息=邻居特征×归一化权重
    
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

def get_A_r(adj, r):
    adj_label = adj
    for i in range(r - 1):
        adj_label = adj_label @ adj #矩阵乘法
    return adj_label

def to_edge_set(edge_index, directed=False):
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        edges = set()
        for s, d in zip(src, dst):
            if not directed:  # 无向图时排序节点对
                s, d = (s, d) if s <= d else (d, s)
            edges.add((s, d))
        return edges

def remove_edges_adj_matrix(adj_matrix, n_remove):
    
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

    return new_adj

def to_edge_index(adj):
    edge_index_all = dense_to_sparse(adj)
    edge_index = edge_index_all[0]
    edge_weight = edge_index_all[1]
    return edge_index, edge_weight

def to_adj(edge_index, edge_weight, x):
    adj = coo_matrix((edge_weight.cpu().detach().numpy(), edge_index.cpu().detach().numpy()), shape=(x, x))
    adj = torch.tensor(adj.toarray(), device='cuda:0', dtype=torch.float32)
    return adj


def cosine_error_loss(x, y, alpha=1):
    # x = F.normalize(x, p=2, dim=-1)
    # x = F.softmax(x, dim=1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    # loss=torch.exp((x * y).sum(dim=-1)-1)
    loss = loss.mean()
    return loss


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


# def accuracy(output, labels):
#     if not hasattr(labels, '__len__'):
#         labels = [labels]
#     if type(labels) is not torch.Tensor:
#         labels = torch.LongTensor(labels)
#     preds = output.max(1)[1].type_as(labels)
#     correct = preds.eq(labels).double()
#     correct = correct.sum()
#     return correct / len(labels)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, self_loop=True, device=None):
        super(GCN, self).__init__()

        self.device = device
        self.gc1 = GCNConv(nfeat, nhid, bias=True, add_self_loops=self_loop)
        self.gc2 = GCNConv(nhid, nclass, bias=True, add_self_loops=self_loop)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x1 = F.relu(self.gc1(x, edge_index, edge_weight))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = self.gc2(x1, edge_index, edge_weight)
        # PyTorch内置函数
        # x1=x1*100
        # x1 = F.softmax(x1, dim=1)
        x1 = F.normalize(x1, p=2, dim=-1)
        return x1

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

class GCN_LP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # z所有节点的表示向量
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print(dst.size())   # (7284, 64)
        r = (src * dst).sum(dim=-1)
        # print(r.size())   (7284)
        return r

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class unnamed_model3(nn.Module):
    def __init__(self, args, input_size, hidden_size, output_size, num_classes):
        super(unnamed_model3, self).__init__()
        self.args = args
        self.best_acc_pred_val1 = 0
        self.best_acc_pred_val2 = 0
        self.label_smoothing = None
        self.forget_rate = 0.9

    def find_neighbour(self, adj, node_list):
        G = nx.from_numpy_array(adj.cpu().detach().numpy())  # 从数组取出元素生成图,可用nx.from_numpy_matrix()
        G.edges(data=True)
        neighbour = set()
        for node in node_list.cpu().detach().numpy().tolist():
            list_node = list(nx.neighbors(G, node))  # find 1_th neighbors
            for i in list_node:
                neighbour.add((node, i))
        return neighbour

    def select_structure(self, y_1, t, forget_rate):

        confidence = y_1.max(1)[0]
        ind_2_sorted = np.argsort(confidence.cpu().data).cuda()
        loss_2_sorted = confidence[ind_2_sorted]
        # mask=torch.where(loss_2_sorted >= 0.9).detach()

        # loss_2 = F.cross_entropy(y_2, t, reduce=False)
        # ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
        # loss_2_sorted = loss_2[ind_2_sorted]
        remember_rate2 = 1 - forget_rate
        num_remember2_1 = int(remember_rate2 * len(loss_2_sorted))
        # num_remember2_2 = mask.shape
        ind_2_update = ind_2_sorted[:num_remember2_1]
        return ind_2_update

    def select_feature(self, recon, x, forget_rate):
        loss = F.mse_loss(recon, x, reduce=False)
        loss = torch.sum(loss, dim=1)
        ind_sorted = np.argsort(loss.cpu().data).cuda()
        loss_sorted = loss[ind_sorted]
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[num_remember:]
        return ind_update,loss

    def select_label(self, y_1, t, train_mask, forget_rate):
        loss_1 = F.cross_entropy(y_1[train_mask], t[train_mask], reduce=False)
        ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
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
        return ind_1_update_final

    # def select_label(self, y_1, t, train_mask, thre):
    #     loss_1 = F.cross_entropy(y_1[train_mask], t[train_mask], reduce=False)
    #
    #     mask=loss_1.cpu().data<thre
    #     ind_1_update_final = train_mask[mask]
    #     return ind_1_update_final


    def fs_c(self, model, features, edge_index,edge_weight,feature_recon,features_old,v,epoch):

        # 学习weieght
        #knn利用feature的消融实验
        # representations, rec_loss_s, y = model(edge_index,self.args.features, self.features_sim, self.label_smoothing)
        representations, rec_loss_s, y = model(edge_index,features, self.features_sim, self.label_smoothing)
        # 融合结构
        ones_matrix1 = torch.ones(self.pred_edge_index1.shape[1]) 
        ones_matrix2 = torch.ones(self.pred_edge_index2.shape[1]) 
        adj_knn1=to_adj(self.pred_edge_index1,ones_matrix1,features.shape[0])
        adj_knn2=to_adj(self.pred_edge_index2,ones_matrix2,features.shape[0])
        adj_fusion=self.fusion_adj(adj_knn1,adj_knn2,1)
        pred_index,_=to_edge_index(adj_fusion)

        predictor_weights = model.get_estimated_weigths(pred_index.long(), representations)
        # predictor_weights_plus = model.get_estimated_weigths(self.pred_edge_index2, representations)
        edge_remain_idx = torch.where(predictor_weights != 0)[0].detach()
        # edge_add=torch.where(predictor_weights_plus >= 0.95)[0].detach()
        predictor_weights = predictor_weights[edge_remain_idx]
        pred_edge_index = pred_index[:, edge_remain_idx]
        # filepath1 = r'./structure_ori/{}.csv'.format(epoch)
        # df1=pd.DataFrame(edge_remain_idx.cpu().detach().numpy())
        # df1.to_csv(filepath1, index=False)
        
        tmp_edge = tu.remove_self_loops(edge_index=edge_index)[0]
        
        


        
        adj = dgl.graph((tmp_edge[0], tmp_edge[1]), num_nodes=features.shape[0])
        adj = dgl.add_self_loop(adj)

        # 融合feature
        features = torch.tensor(features, device='cuda:0')
        features_fusion=self.fusion_feature(self.args.features,features,0)

        rec_loss_f, recon = feature_recon(adj.cpu(),predictor_weights.cpu(),features_fusion.cpu())
        nodes,loss_f = self.select_feature(recon.cuda(), self.features, self.args.feature_noise)#对于原始feature loss较高
        filepath1 = r'./feature_truth/nodes.csv'
        df1= pd.DataFrame(nodes.cpu().detach().numpy())
        df1.to_csv(filepath1, index=False)
        # set1=set(nodes1)
        # set2=set(nodes2)
        # nodes=list(set2-set1)
        features_after = features.clone()
        
        recon=torch.tensor(recon,device=self.args.device).clone()
        # aggregator = WeightedAggregation()
        # self.best_edge_idx1=self.edge_index.clone()
        # self.best_features1 = self.features1.clone()
        # self.best_pred_graph1 =self.edge_weight.clone()
        # feats = aggregator(self.best_features1, self.best_edge_idx1, self.best_pred_graph1)
        feats = scatter(features[tmp_edge[1]], tmp_edge[0], dim=0, dim_size=features.size(0), reduce='mean')
        # feats = aggregator(features, pred_edge_index, predictor_weights)
        #不更新features注释此行
        features_after[nodes] = feats[nodes].detach()
        # recon_l=F.mse_loss(repr1,features)
        # consistency_loss=0
        return features_after.cuda(), pred_edge_index.cuda(), predictor_weights.cuda(), rec_loss_s, rec_loss_f, representations,nodes,loss_f
        # return features_after.cuda(), self.args.edge_index.cuda(), self.args.edge_weight, rec_loss_s, rec_loss_f, representations,nodes,loss_f
        # return features.cuda(), pred_edge_index.cuda(), predictor_weights.cuda(), rec_loss_s, rec_loss_f, representations,nodes,loss_f

    def generate_loss_matrix(self,losses):
        """
        生成上三角loss和矩阵（仅保留i<j的组合，对角线置零）
        :param losses: 每个点的loss值列表
        :return: 上三角loss和矩阵
        """
        losses = np.array(losses.cpu().detach())
        # 生成i<j的上三角矩阵，其他位置自动置零
        return np.triu(losses[:, np.newaxis] + losses[np.newaxis, :], k=1)

    def get_top_coordinates(self,mat, percent=0.3):
        """
        获取非零元素中最大的percent比例的坐标
        :param mat: 上三角loss和矩阵
        :param percent: 比例（0-1）
        :return: 降序排列的坐标列表
        """
        # 获取所有非零元素的坐标
        coords = np.argwhere(mat > 0)
        if len(coords) == 0:
            return []
        values = mat[coords[:, 0], coords[:, 1]]

        # 计算选取数量
        k = int(np.ceil(percent * len(coords)))
        if k == 0:
            return []

        # 高效获取topk索引并排序
        indices = np.argpartition(-values, k - 1)[:k]
        sorted_indices = indices[np.argsort(-values[indices])]

        return [tuple(coords[i]) for i in sorted_indices]

    def remove_edges_from_adjacency(self,adj_matrix, edges_to_remove):
        """
        从邻接矩阵中删除指定的边
        :param adj_matrix: 图的邻接矩阵
        :param edges_to_remove: 要删除的边的坐标列表，例如[(i1, j1), (i2, j2), ...]
        :return: 更新后的邻接矩阵
        """
        # 复制邻接矩阵以避免修改原始数据
        updated_adj_matrix = adj_matrix.clone()

        # 遍历要删除的边，将对应位置的值设为0
        for i, j in edges_to_remove:
            updated_adj_matrix[i, j] = 0
            updated_adj_matrix[j, i] = 0  # 如果是无向图，需要同时删除对称位置

        return updated_adj_matrix

    def fit(self):
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
        self.args.drop_edge_rate = 0#随机mask边
        self.args.replace_rate = 0.0
        self.args.activation = "prelu"
        self.args.loss_fn = "sce"
        self.args.alpha_l = 2
        self.args.concat_hidden = False
        self.args.num_features = self.args.features.shape[1]
        args = self.args

        self.edge_index = args.edge_index
        self.edge_weight=args.edge_weight
        self.best_edge_idx1=self.edge_index.clone()
        self.best_features1 = self.args.features1.clone()
        self.best_pred_graph1 =self.edge_weight.clone()

        self.features = args.features.clone()
        self.labels = args.labels.clone()
        self.idx_train = args.train_index
        self.idx_val = args.val_index
        self.idx_test = args.test_index
        self.device = args.device
        self.labels_noise=args.labels_noise

        self.q_phi1 = q_phi1(self.features, self.features.shape[1], args, device=self.device).to(self.device)
        self.q_phi3 = GCN(nfeat=self.features.shape[1],
                          nhid=self.args.hidden_size,
                          nclass=self.labels.max().item() + 1,
                          dropout=self.args.dropout, device=self.device).to(self.device)
        # self.q_phi2 = q_phi1(self.features, self.features.shape[1], args, device=self.device).to(self.device)

        # self.q_phi4 = GCN(nfeat=self.features.shape[1],
        #                   nhid=self.args.hidden_size,
        #                   nclass=self.labels.max().item() + 1,
        #                   dropout=self.args.dropout, device=self.device).to(self.device)
        self.feature_recon1 = build_model(self.args)
        # self.feature_recon2 = build_model(self.args)
        self.fusion_adj = DimWiseFusion(self.args.adj.shape).to(self.device)
        self.fusion_feature = DimWiseFusion(self.args.features.shape).to(self.device)
       
        self.features_sim = F.normalize(self.features, dim=1, p=2).mm(
            F.normalize(self.features, dim=1, p=2).T).fill_diagonal_(0.0)
        
        
        # Prior of Z_A
        if self.args.K > 0:
            if self.args.hop == 0:
                feats = self.features.clone()
            elif self.args.hop > 0:
                tmp_edge = tu.add_self_loops(edge_index=self.edge_index, num_nodes=self.features.shape[0])[0]
                feats = scatter(self.features[tmp_edge[1]], tmp_edge[0], dim=0, dim_size=self.features.size(0),
                                reduce='mean')
            sim = F.normalize(feats, dim=1, p=2).mm(F.normalize(feats, dim=1, p=2).T).fill_diagonal_(0.0)
            dst = sim.topk(self.args.K, 1)[1].to(self.device)
            src = torch.arange(sim.size(0)).unsqueeze(1).expand_as(dst).to(self.device)
            self.knn_edge = torch.stack([src.reshape(-1), dst.reshape(-1)])
            self.pred_edge_index1 = tu.to_undirected(self.knn_edge).to(self.device)
            # self.pred_edge_index =edge_index.clone()
        else:
            self.pred_edge_index1 = self.edge_index.clone()
        self.pred_edge_index1 = tu.coalesce(
            torch.cat([self.edge_index.to(self.device), self.pred_edge_index1.to(self.device)], dim=1))
        self.pred_edge_index2= self.pred_edge_index1.clone()
        #without knn_graph
        # self.pred_edge_index1=self.edge_index.clone
        
        self.optimizer1 = optim.Adam(list(self.q_phi1.parameters()) + list(self.q_phi3.parameters())+list(self.feature_recon1.parameters())+list(self.fusion_adj.parameters())+list(self.fusion_feature.parameters()), lr=args.lr, weight_decay=args.wd)
        # self.optimizer2 = optim.Adam(list(self.q_phi2.parameters()) + list(self.q_phi4.parameters())+list(self.feature_recon2.parameters()), lr=args.lr, weight_decay=args.wd)
        
        cnt_wait = 0
        acc_list=[]
        str_loss_list=[]
        fea_loss_list=[]
        training_time_list=[]
        for epoch in range(args.epochs):
            val_acc,str_loss,fea_loss,t = self.train(epoch, self.features, self.edge_index,self.edge_weight, self.idx_train, self.idx_val, self.idx_test)
            acc_list.append(val_acc.item())
            str_loss_list.append(str_loss.item())
            fea_loss_list.append(fea_loss.item())
            training_time_list.append(t)
            if val_acc >= self.best_acc_pred_val1:
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                print('Early stopping!')
                break
        print("training_time:{:.4f}".format(np.mean(np.array(training_time_list))),"{:.4f}".format(np.std(np.array(training_time_list))))
        # epochs = range(1, len(acc_list)+1)

        # # 绘制准确率曲线
        # plt.figure(1)
        # plt.plot(epochs, str_loss_list, 'b', label='str_loss')

        # # 添加标题和标签
        # plt.title('str_loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('loss')
        # plt.legend()

        # # 显示图形
        # # plt.show()
        # plt.savefig('structure_loss.png', dpi=300, bbox_inches='tight')

        # plt.figure(2)
        # plt.plot(epochs, fea_loss_list, 'b', label='fea_loss')

        # # 添加标题和标签
        # plt.title('fea_loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('loss')
        # plt.legend()

        # # 显示图形
        # # plt.show()
        # plt.savefig('feature_loss.png', dpi=300, bbox_inches='tight')

        print("Optimization Finished!")

        print("picking the best model according to validation performance")

        self.q_phi3.load_state_dict(self.predictor_model_weigths1)
        # self.q_phi4.load_state_dict(self.predictor_model_weigths2)
        # if not os.path.exists('./teacher_models/{}/g_{}_l_{}_f_{}'.format(args.dataset,args.graph_noise,args.label_noise,args.feature_noise)):
        #                 os.mkdir('./teacher_models/{}/g_{}_l_{}_f_{}'.format(args.dataset,args.graph_noise,args.label_noise,args.feature_noise))
        
        save_path = f"./saved_model_weights/{self.args.dataset}.pth"

        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存模型权重
        torch.save(self.predictor_model_weigths1, save_path)
        print(f"model weights saved: {save_path}")

    def train(self, epoch, features, edge_index,edge_weight, train_mask, val_mask, test_mask):
        self.q_phi1.train()
        # self.q_phi2.train()
        self.q_phi3.train()
        # self.q_phi4.train()
        self.feature_recon1.train()
        # self.feature_recon2.train()
      
        
        t = time.time()
        args = self.args

        flag = 1
        self.optimizer1.zero_grad()
        # self.optimizer2.zero_grad()
       
        # 进行feature和structure的净化
        args.features1, self.pred_edge_index1, predictor_weights1, rec_loss_s1, rec_loss_f1, representations1,nodes1,loss_f1 = self.fs_c(
            self.q_phi1, args.features1, edge_index,edge_weight, self.feature_recon1,args.features_old_ind1,1,epoch)
        # args.features2, pred_edge_index2, predictor_weights2, rec_loss_s2, rec_loss_f2, representations2,nodes2,loss_f2 = self.fs_c(
        #     self.G,self.q_phi2, args.features2, edge_index, self.feature_recon2,args.features_old_ind2,2)
        
        pred_edge_index1=self.pred_edge_index1
        features1=args.features1
        # 使用coteaching选取高置信度样本
        output_z_y1 = self.q_phi3(features1.data, pred_edge_index1.data, predictor_weights1.data)
        # output_z_y2 = self.q_phi4(args.features2.data, pred_edge_index2.data, predictor_weights2.data)
        
        if self.args.teacher_student=='student':
            if args.teacher_student=='student':
                path='./teacher_models/'+args.noise_type+'/'+args.dataset
            with open(path+'/g_{}_l_{}_f_{}/soft_label2020.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise), 'rb') as f:
                args.soft_label1 = pickle.load(f)
            with open(path+'/g_{}_l_{}_f_{}/soft_label2021.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise), 'rb') as f:
                args.soft_label2 = pickle.load(f)
            with open(path+'/g_{}_l_{}_f_{}/soft_label2022.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise), 'rb') as f:
                args.soft_label3 = pickle.load(f)
            with open(path+'/g_{}_l_{}_f_{}/soft_label2023.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise), 'rb') as f:
                args.soft_label4 = pickle.load(f)
            with open(path+'/g_{}_l_{}_f_{}/soft_label2024.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise), 'rb') as f:
                args.soft_label5 = pickle.load(f)
            with open(path+'/g_{}_l_{}_f_{}/soft_label2025.pkl'.format(args.graph_noise,args.label_noise,args.feature_noise), 'rb') as f:
                args.soft_label6 = pickle.load(f)
            label_index1 = self.select_label(self.args.soft_label1, self.args.targets_oneHot1, args.train_index_clean,args.label_noise)
            label_index2 = self.select_label(self.args.soft_label2, self.args.targets_oneHot1, args.train_index_clean,args.label_noise)
            label_index3 = self.select_label(self.args.soft_label3, self.args.targets_oneHot1, args.train_index_clean,args.label_noise)
            label_index4 = self.select_label(self.args.soft_label4, self.args.targets_oneHot1, args.train_index_clean,args.label_noise)
            label_index5 = self.select_label(self.args.soft_label5, self.args.targets_oneHot1, args.train_index_clean,args.label_noise)
            label_index6 = self.select_label(self.args.soft_label6, self.args.targets_oneHot1, args.train_index_clean,args.label_noise)
            intersection = label_index1
            intersection = tensor_intersection(intersection, label_index2)
            intersection = tensor_intersection(intersection, label_index3)
            intersection = tensor_intersection(intersection, label_index4)
            intersection = tensor_intersection(intersection, label_index5)
            intersection = tensor_intersection(intersection, label_index6)
            # intersection = torch.intersection(label_index1,label_index2,label_index3,label_index4)
            # print("intersection",intersection)
            # print(intersection)
            # if epoch==0 and args.teacher_student=="student":
            #     clean_sample=args.labels_noise==args.clean_labels
            #     indices=torch.nonzero(clean_sample)
            #     label_index1=tensor_intersection(label_index1,indices)
            #     label_index2=tensor_intersection(label_index2,indices)
            #     label_index3=tensor_intersection(label_index3,indices)
            #     label_index4=tensor_intersection(label_index4,indices)
                
            #     filepath1 = r'./label/label1.csv'
            #     df1=pd.DataFrame(label_index1.cpu().detach().numpy())
            #     df1.to_csv(filepath1, index=False)
            #     filepath2 = r'./label/label2.csv'
            #     df2=pd.DataFrame(label_index2.cpu().detach().numpy())
            #     df2.to_csv(filepath2, index=False)
            #     filepath3 = r'./label/label3.csv'
            #     df3=pd.DataFrame(label_index3.cpu().detach().numpy())
            #     df3.to_csv(filepath3, index=False)
            #     filepath4 = r'./label/label4.csv'
            #     df4=pd.DataFrame(label_index4.cpu().detach().numpy())
            #     df4.to_csv(filepath4, index=False)
            #     filepath5 = r'./label/commen.csv'
            #     intersection1=tensor_intersection(intersection,indices)
            #     df5=pd.DataFrame(intersection1.cpu().detach().numpy())
            #     df5.to_csv(filepath5, index=False)
                
            #     filepath6 = r'./label/truth.csv'
            #     df6=pd.DataFrame(clean_sample.cpu().detach().numpy())
            #     df6.to_csv(filepath6, index=False)
               
                
            log_p = F.log_softmax(output_z_y1, dim=1)
            q1 = F.softmax(self.args.soft_label1, dim=1)
            q2 = F.softmax(self.args.soft_label2, dim=1)
            q3 = F.softmax(self.args.soft_label3, dim=1)
            q4 = F.softmax(self.args.soft_label4, dim=1)
            q5 = F.softmax(self.args.soft_label5, dim=1)
            q6 = F.softmax(self.args.soft_label6, dim=1)
            kld1=F.kl_div(log_p,q1)
            kld2=F.kl_div(log_p,q2)
            kld3=F.kl_div(log_p,q3)
            kld4=F.kl_div(log_p,q4)
            kld5=F.kl_div(log_p,q5)
            kld6=F.kl_div(log_p,q6)
        # label_index2 = self.select_label(output_z_y2, self.args.targets_oneHot2, args.train_index_clean,args.rate_schedule[epoch])
        # print(label_index1.shape)
        if self.args.teacher_student=='student':
            loss1 = cosine_error_loss(output_z_y1[intersection], args.targets_oneHot1[intersection])
            loss_1=F.cross_entropy(output_z_y1[train_mask], args.targets_oneHot1[train_mask],reduce=False)
        else:
            # loss1 =cosine_error_loss(output_z_y1[train_mask], args.targets_oneHot1[train_mask])
            loss1 =cosine_error_loss(output_z_y1[train_mask], args.targets_oneHot1[train_mask])
            loss_1=F.cross_entropy(output_z_y1[train_mask], args.targets_oneHot1[train_mask],reduce=False)
        

        #根据loss筛选下一轮需要删除的边和不需要重构的features
        # loss_matrix1 = self.generate_loss_matrix(loss_1)
        # self.top_coords1 = self.get_top_coordinates(loss_matrix1, self.args.graph_noise*2/3)
        # loss_matrix2 = self.generate_loss_matrix(loss_2)
        # self.top_coords2 = self.get_top_coordinates(loss_matrix2, self.args.graph_noise*2/3)
        # args.features_old_ind2=self.select_label(output_z_y1, self.args.labels_oneHot, args.train_index_clean,args.feature_noise)
        # args.features_old_ind1 = self.select_label(output_z_y2, self.args.labels_oneHot, args.train_index_clean,args.feature_noise)

        # print(args.features_old_ind1.shape)
       
        # print(kld1)
        if self.args.teacher_student=='student':
            # loss_all1 = loss1 + 0.03 * rec_loss_s1 + rec_loss_f1 +(kld1+kld2+kld3+kld4)*300
            # if self.args.ntea == 4:
            loss_all1 = loss1 + 0.03 * rec_loss_s1 + rec_loss_f1 +(kld1+kld2+kld3+kld4+kld5+kld6)*self.args.lambda_kld
        else:
            loss_all1 = loss1 + 0.03 * rec_loss_s1 + rec_loss_f1 
        # loss_all2 = loss2 + 0.03 * rec_loss_s2 + rec_loss_f2 
        loss_all1.backward()
        self.optimizer1.step()
        # loss_all2.backward()
        # self.optimizer2.step()
        self.q_phi3.eval()
        output0 = self.q_phi3(features1, pred_edge_index1, predictor_weights1)
        acc_pred_train0 = accuracy(output0[train_mask], args.targets_c_oneHot[train_mask])
        acc_pred_val0 = accuracy(output0[val_mask], args.targets_c_oneHot[val_mask])
        acc_pred_test0 = accuracy(output0[test_mask], args.targets_c_oneHot[test_mask])

        adj_pred=to_adj(pred_edge_index1,predictor_weights1,features1.shape[0])
        str_loss = F.mse_loss(self.args.clean_adj, adj_pred)
        fea_loss = F.mse_loss(self.args.clean_features, features1)
        # if epoch%20==0:
        #     filepath1 = r'./structure_/str_{}.csv'.format(epoch)
        #     df1=pd.DataFrame(adj_pred.cpu().detach().numpy())
        #     df1.to_csv(filepath1, index=False)
        #     filepath2 = r'./feature_/fea_{}.csv'.format(epoch)
        #     df2=pd.DataFrame(args.features1.cpu().detach().numpy())
        #     df2.to_csv(filepath2, index=False)


        pre_value_max, pre_index_max = output_z_y1.max(1)
        t_time=time.time() - t
        # print(pre_value_max[0])
        if flag == 0:
            path='./label_analyze3/'
            if not os.path.exists(path+'epoch{}'.format(epoch)):
                os.mkdir(path+'epoch{}'.format(epoch))
            with open(path+'epoch{}/score_label.pkl'.format(epoch), 'wb') as f:
                pickle.dump(pre_value_max, f)  # 分数
            with open(path+'epoch{}/pseudo_label.pkl'.format(epoch), 'wb') as f:
                pickle.dump(pre_index_max, f)  # 伪标签
            loss_1 = []
            for mask in range(output_z_y1.shape[0]):
                loss_1.append(
                    cosine_error_loss(output_z_y1[mask], self.args.targets_oneHot1[mask]).cpu().detach().numpy())
            with open(path+'epoch{}/loss_label.pkl'.format(epoch), 'wb') as f:
                pickle.dump(loss_1, f)  # loss_label
            with open(path+'epoch{}/loss_feature.pkl'.format(epoch), 'wb') as f:
                pickle.dump(loss_f1, f)  # loss_feature
            with open(path+'epoch{}/nodes.pkl'.format(epoch), 'wb') as f:
                mask1 = torch.zeros(self.args.labels.shape[0])
                mask1[nodes1] = 1
                mask1 = torch.tensor(mask1, device='cuda:0')
                pickle.dump(mask1, f)  # 修改feature的点
            with open(path+'epoch{}/label_select.pkl'.format(epoch), 'wb') as f:
                mask2 = torch.zeros(self.args.labels.shape[0])
                mask2[label_index1] = 1
                mask2 = torch.tensor(mask2, device='cuda:0')
                pickle.dump(mask2, f)  # 选取的label
            with open(path+'epoch{}/remove_edge.pkl'.format(epoch), 'wb') as f:
                pickle.dump(self.top_coords1, f)  # 需要删除的edge
            with open(path+'epoch{}/recon_feature.pkl'.format(epoch), 'wb') as f:
                mask2 = torch.zeros(self.args.labels.shape[0])
                mask2[args.features_old_ind1] = 1
                mask2 = torch.tensor(mask2, device='cuda:0')
                pickle.dump(mask2, f)  # 不需要recon的点
        if acc_pred_val0 >= self.best_acc_pred_val1:
            print(1)
            self.best_acc_pred_val1 = acc_pred_val0
            self.best_features1 = features1.detach()
            self.best_pred_graph1 = predictor_weights1.detach()
            self.best_edge_idx1 = pred_edge_index1.detach()
            self.predictor_model_weigths1 = deepcopy(self.q_phi3.state_dict())
            better_inex = 1
        


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

        return acc_pred_val0,str_loss,fea_loss,t_time

    def test(self, idx_test,seed,noise_type):
        self.q_phi3.eval()
        estimated_weights1 = self.best_pred_graph1
        pred_edge_index1 = self.best_edge_idx1
        features1 = self.best_features1
        t1= time.time()
        output0 = self.q_phi3(features1, pred_edge_index1, estimated_weights1)
        t2 = time.time()
        # self.q_phi4.eval()
        # estimated_weights2 = self.best_pred_graph1
        # pred_edge_index2 = self.best_edge_idx1
        # features2 = self.best_features1
        # output1 = self.q_phi4(features2, pred_edge_index2, estimated_weights2)
        if self.args.teacher_record and self.args.teacher_student=='teacher':
            if not os.path.exists('./teacher_models/{}/{}/g_{}_l_{}_f_{}'.format(noise_type,self.args.dataset,self.args.graph_noise,self.args.label_noise,self.args.feature_noise)):
                            os.makedirs('./teacher_models/{}/{}/g_{}_l_{}_f_{}'.format(noise_type,self.args.dataset,self.args.graph_noise,self.args.label_noise,self.args.feature_noise))
            with open('./teacher_models/{}/{}/g_{}_l_{}_f_{}/soft_label{}.pkl'.format(noise_type,self.args.dataset,self.args.graph_noise,self.args.label_noise,self.args.feature_noise,seed), 'wb') as f:
                    pickle.dump(output0, f)
        # with open('./teacher_models/{}/g_{}_l_{}_f_{}/soft_label4.pkl'.format(self.args.dataset,self.args.graph_noise,self.args.label_noise,self.args.feature_noise), 'wb') as f:
        #         pickle.dump(output1, f)
        acc_pred_test0 = accuracy(output0[idx_test], self.args.targets_c_oneHot[idx_test]).item()
        pre,rec,f1=calculate_classification_metrics(output0[idx_test], self.args.targets_c_oneHot[idx_test])
        print("Test Accuray: #1 = %f" % (acc_pred_test0))
        return acc_pred_test0, output0.detach(),pre,rec,f1,t2-t1


class q_phi1(nn.Module):

    def __init__(self, features, nfea, args, device='cuda'):
        super(q_phi1, self).__init__()
        self.estimator = GCN(nfea, args.hidden_size, args.hidden_size, dropout=0.0, device=device)
        self.device = device
        self.args = args
        self.y_fc = nn.Linear(args.hidden_size, args.num_classes)
        

    def forward(self, edge_index, features, sim, label_smoothing):
        representations = self.estimator(features, edge_index, \
                                         torch.ones([edge_index.shape[1]]).to(self.device).float())
        representations = F.normalize(representations, dim=-1)
        # print(representations.shape)
        y = self.y_fc(representations)
        rec_loss, pos_w = self.reconstruct_loss(edge_index, representations, sim, label_smoothing)
        return representations, rec_loss, y

    def get_estimated_weigths(self, edge_index, representations):
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0, x1), dim=1)
        estimated_weights = F.relu(output)
        # estimated_weights = torch.cat([x0, x1], dim=1)
        if estimated_weights.shape[0] != 0:
            estimated_weights = estimated_weights.masked_fill(estimated_weights < self.args.tau, 0.0)
        # self.mlp = nn.Sequential(
        #     nn.Linear(estimated_weights.shape[1], 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # ).to(device='cuda:0')
        return estimated_weights
        ghts
    def reconstruct_loss(self, edge_index, representations, sim, label_smoothing):
        num_nodes = representations.shape[0]
        randn = torch_geometric.utils.negative_sampling(edge_index, num_nodes=num_nodes,
                                                        num_neg_samples=edge_index.size(1))
        randn = randn[:, randn[0] < randn[1]]

        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg_w = F.relu(torch.sum(torch.mul(neg0, neg1), dim=1))

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos_w = F.relu(torch.sum(torch.mul(pos0, pos1), dim=1))

        pos_fsim = sim[edge_index[0], edge_index[1]]

        pos = pos_fsim * self.args.theta + pos_w * (1 - self.args.theta)

        if label_smoothing is None:
            rec_loss = (F.mse_loss(neg_w, torch.zeros_like(neg_w), reduction='sum') \
                        + F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) \
                       * num_nodes / (randn.shape[1] + edge_index.shape[1])

        else:
            rec_loss = (F.mse_loss(neg_w, torch.zeros_like(neg_w), reduction='sum') \
                        + F.mse_loss(pos, label_smoothing, reduction='sum')) \
                       * num_nodes / (randn.shape[1] + edge_index.shape[1])

        return rec_loss, pos_w

