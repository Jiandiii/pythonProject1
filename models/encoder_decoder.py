import torch
import torch.nn as nn
from models.gnn_models import *
import copy
from models.Semi_RNCGLN import *
from torch_geometric.utils import to_undirected, dense_to_sparse, to_dense_adj, is_undirected, remove_self_loops, degree, to_scipy_sparse_matrix, from_scipy_sparse_matrix
from scipy import sparse


def get_feature_dis_after(x,k,adj):
    # sim=F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
    sim=F.relu(x@x.T)
    sim=sim*adj
    indices = np.argpartition(-sim.cpu().detach().numpy().ravel(), k)[:k]
    indices = torch.tensor(indices, device='cuda:0')
    x_dis=torch.zeros(sim.shape[0],sim.shape[0])
    x_dis = torch.tensor(x_dis, device='cuda:0')
    x_dis[indices // x.shape[0], indices % x.shape[0]] = sim[indices // x.shape[0], indices % x.shape[0]]
    # x_dis = sim.masked_fill(sim < 0.01, 0.0)
    return x_dis

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

def get_estimated_weigths(edge_index, representations,tau):
    x0 = representations[edge_index[0]]
    x1 = representations[edge_index[1]]
    output = torch.sum(torch.mul(x0, x1), dim=1)
    estimated_weights = F.relu(output)
    if estimated_weights.shape[0] != 0:
        estimated_weights = estimated_weights.masked_fill(estimated_weights < torch.mean(estimated_weights)*tau, 0.0)
        # estimated_weights = estimated_weights.masked_fill(estimated_weights < tau, 0.0)

    return estimated_weights

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model


        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class H_ENCODER(nn.Module):
    def __init__(self,args,num_class, input_size = 2, hidden_size = 5, output_size=128):
        super().__init__()
        self.args=args
        # self.gnn=GAT(input_size=input_size,hidden_size=hidden_size,output_size=output_size,dropout=0.6,alpha=0.2,nheads=8)
        self.gnn = GCN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=0.6)
        self.h_fc_mu = nn.Linear(hidden_size*2, output_size)  # fc21 for mean of h
        self.h_fc=nn.Linear(input_size, input_size)
        self.h_fc_logvar = nn.Linear(output_size*2, output_size)
        self.y_fc_emb=nn.Linear(num_class,output_size)
        self.output_size=output_size
        self.y_fc = nn.Linear(hidden_size,num_class)
        self.norm=Norm(input_size)
    def forward(self,features,adj):
        #print(adj)
        h=F.normalize(self.gnn(F.normalize(features),adj))
        y=F.softmax(self.y_fc(h))
        return h,y

class Y_ENCODER(nn.Module):
    def __init__(self,hidden_size,num_classes,input_size):
        super().__init__()
        # self.gnn=GAT(input_size=input_size,hidden_size=hidden_size,output_size=num_classes,dropout=0.6,alpha=0.2,nheads=8)
        self.gnn = GCN(input_size=input_size, hidden_size=hidden_size, output_size=num_classes, dropout=0.6)
        self.linear = nn.Linear(hidden_size,num_classes)
    def forward(self,x,adj_c):
        # edge_index = dense_to_sparse(adj_c)
        # y = self.gnn(x, edge_index[0], edge_index[1])
        y=F.softmax(self.gnn(F.normalize(x),adj_c))
        # y=self.linear(h)
        # y = F.softmax(y,dim=1)
        return y

class Y_PRETRAINED(nn.Module):
    def __init__(self,data,g,l,f):
        super().__init__()
        self.model=torch.load('./pretrained/{}_g{}_l{}_f{}.pth'.format(data, g, l,f))
    def forward(self,features,adj):
        model=self.model
        model.eval()
        y_bar=model(features,adj)
        return y_bar

class Z_ENCODER(nn.Module):
    def __init__(self, args,input_size = 2708, hidden_size = 128, output_size=128,num_class=7):
        super().__init__()
        # self.gnn=GAT(input_size=input_size,hidden_size=hidden_size,output_size=output_size,dropout=0.6,alpha=0.2,nheads=1)
        self.gnn = GCN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=0.6)
        self.y_fc_emb = nn.Linear(num_class, output_size)
        self.out_fc = nn.Linear(output_size*2, output_size)
        self.z_fc_mu = nn.Linear(hidden_size, args.z_dim)  # fc21 for mean of Z
        self.z_fc_logvar = nn.Linear(hidden_size, args.z_dim)  # fc22 for log variance of Z

    def forward(self, x, adj,y):
        # edge_index = dense_to_sparse(adj)
        # h = self.gnn(x, edge_index[0], edge_index[1])
        h=self.gnn(F.normalize(x),adj)
        y_emb = self.y_fc_emb(y)
        out=torch.cat([h, y_emb], 1)
        out=self.out_fc(out)
        mu = F.elu(self.z_fc_mu(out))
        logvar = F.elu(self.z_fc_logvar(out))
        return mu, logvar
class X_DECODER(nn.Module):
    def __init__(self,args, num_class,input_size=2,hidden_size=5,output_size=128):
        super().__init__()
        self.output_size=output_size
        self.y_fc_emb = nn.Linear(num_class, input_size)
        self.z_fc=nn.Linear(args.z_dim,input_size)
        self.x_hat_fc=nn.Linear(input_size*2,input_size)
        self.args=args
    def forward(self,z,y):
        # h=self.h_fc(h)
        x=self.z_fc(z)
        y=torch.tensor(y,device=self.args.device)
        y_emb=self.y_fc_emb(y)
        x_hat_after = torch.cat([y_emb, x], 1)
        x_hat_after=self.x_hat_fc(x_hat_after)
        return x_hat_after
class Y_DECODER(nn.Module):
    def __init__(self,num_class,input_size=2,hidden_size=5,output_size=128):
        super().__init__()
        # self.gnn = GAT(input_size=input_size,hidden_size=hidden_size,output_size=output_size,dropout=0.6,alpha=0.2,nheads=1)
        self.gnn = GCN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=0)
        self.output_size = output_size
        self.y_fc_emb = nn.Linear(num_class, output_size)
        self.y_fc = nn.Linear(hidden_size*2, num_class)
    def forward(self,features,adj,y):
        # edge_index = dense_to_sparse(adj)
        # h = self.gnn(features, edge_index[0], edge_index[1])
        h = self.gnn(F.normalize(features), adj)
        y_emb=self.y_fc_emb(y)
        y_hat_after = torch.cat([y_emb, h], 1)
        y_hat_after=self.y_fc(y_hat_after)
        y_hat_after=F.softmax(y_hat_after)
        # y_hat_after=self.gnn(features,adj)
        return y_hat_after