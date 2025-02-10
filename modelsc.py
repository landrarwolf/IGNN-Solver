# import ipdb
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
# from functions import *
# from layers import ImplicitGraph, IDM_SGC
# from torch.nn import Parameter
# from utils import get_spectral_rad, SparseDropout
from torch_geometric.nn import GCNConv, GATConv, SGConv, APPNP, GCN2Conv, JumpingKnowledge, MessagePassing
from torch_geometric.utils import remove_self_loops, to_scipy_sparse_matrix

from utils import *


class IDMFunction(Function):
    @staticmethod
    def forward(ctx, X, F, S, Q_S, Lambda_S, gamma):
        # Lambda_F, Q_F = torch.symeig(g(F), eigenvectors=True)
        Lambda_F, Q_F = torch.linalg.eigh(g(F))

        Lambda_F = Lambda_F.view(-1, 1)
        G = get_G(Lambda_F, Lambda_S, gamma)
        Z = Q_F @ (G * (Q_F.t() @ X @ Q_S)) @ Q_S.t()
        ctx.save_for_backward(F, S, Q_F, Q_S, Z, G, X, gamma)
        return Z

    @staticmethod
    def backward(ctx, grad_output):
        grad_Z = grad_output
        F, S, Q_F, Q_S, Z, G, X, gamma = ctx.saved_tensors
        FF = F.t() @ F
        FF_norm = torch.norm(FF, p='fro')
        R = G * (Q_F.t() @ grad_Z @ Q_S)
        R = Q_F @ R @ Q_S.t() @ torch.sparse.mm(S, Z.t())
        scalar_1 = gamma * (1 / (FF_norm + epsilon_F))
        scalar_2 = torch.sum(FF * R)
        scalar_2 = 2 * scalar_2 * (1 / (FF_norm ** 2 + epsilon_F * FF_norm))
        grad_F = (R + R.t()) - scalar_2 * FF
        grad_F = scalar_1 * (F @ grad_F)
        grad_X = None
        return grad_X, grad_F, None, None, None, None


class IDM_SGC(nn.Module):
    def __init__(self, adj, sp_adj, m, num_eigenvec, gamma, adj_preload_file=None):
        super(IDM_SGC, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        self.S = adj
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        sy = (abs(sp_adj - sp_adj.T) > 1e-10).nnz == 0
        if sy:
            self.Lambda_S, self.Q_S = scipy.linalg.eigh(sp_adj.toarray())
        else:
            self.Lambda_S, self.Q_S = scipy.linalg.eig(sp_adj.toarray())
        self.Lambda_S = torch.from_numpy(self.Lambda_S).type(torch.FloatTensor).cuda()
        self.Q_S = torch.from_numpy(self.Q_S).type(torch.FloatTensor).cuda()
        self.Lambda_S = self.Lambda_S.view(-1, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        return IDMFunction.apply(X, self.F, self.S, self.Q_S, self.Lambda_S, self.gamma)


class EIGNN_Linear(nn.Module):
    def __init__(self, adj, sp_adj, m, m_y, num_eigenvec, gamma):
        super(EIGNN_Linear, self).__init__()
        self.EIGNN = IDM_SGC(adj, sp_adj, m, num_eigenvec, gamma)
        self.B = nn.Linear(m, m_y, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.B.reset_parameters()
        self.EIGNN.reset_parameters()

    def forward(self, X, edge_index=None):
        output = self.EIGNN(X).t()
        output = F.normalize(output, dim=-1)
        output = F.dropout(output, 0.5, training=self.training)
        output = self.B(output)
        return output


epsilon_F = 10 ** (-12)


def g(F):
    FF = F.t() @ F
    FF_norm = torch.norm(FF, p='fro')
    return (1 / (FF_norm + epsilon_F)) * FF


def get_G(Lambda_F, Lambda_S, gamma):
    G = 1.0 - gamma * Lambda_F @ Lambda_S.t()
    G = 1 / G
    return G


class GCN(nn.Module):
    def __init__(self, m, m_y, hidden):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(m, hidden)
        self.gc2 = GCNConv(hidden, m_y)

    def forward(self, x, edge_index):
        out = self.gc1(x, edge_index.coalesce())
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.gc2(out, edge_index.coalesce())
        return out


class GAT(nn.Module):
    def __init__(self, m, m_y, hidden, heads):
        super(GAT, self).__init__()
        self.gat1 = GATConv(m, hidden, heads=heads)
        self.gat2 = GATConv(heads * hidden, m_y, heads=heads)

    def forward(self, x, edge_index):
        out = self.gat1(x, edge_index.coalesce())
        out = F.elu(out)
        out = F.dropout(out, p=0.8, training=self.training)
        out = self.gat2(out, edge_index.coalesce())
        return out


class SGC(nn.Module):
    def __init__(self, m, m_y, K):
        super(SGC, self).__init__()
        self.sgc = SGConv(m, m_y, K)
        self.reset_parameters()

    def reset_parameters(self):
        self.sgc.reset_parameters()

    def forward(self, x, edge_index):
        out = self.sgc(x, edge_index.coalesce())
        return out


class APPNP_Net(nn.Module):
    def __init__(self, m, m_y, nhid, K, alpha):
        super(APPNP_Net, self).__init__()
        self.lin1 = nn.Linear(m, nhid)
        self.lin2 = nn.Linear(nhid, m_y)
        self.prop1 = APPNP(K=K, alpha=alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        out = self.prop1(x, edge_index.coalesce())
        return out


class GCN_JKNet(torch.nn.Module):
    def __init__(self, m, m_y, hidden, layers=8):
        in_channels = m
        out_channels = m_y

        super(GCN_JKNet, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        # self.conv1 = GCNConv(in_channels, hidden)
        # self.conv2 = GCNConv(hidden, hidden)
        self.lin1 = nn.Linear(layers * hidden, out_channels)
        # self.lin1 = torch.nn.Linear(64, out_channels)
        # self.one_step = APPNP(K=1, alpha=0)
        # self.JK = JumpingKnowledge(mode='lstm',
        #                            channels=64,
        #                            num_layers=4)
        self.JK = JumpingKnowledge(mode='cat')

    def forward(self, x, edge_index):

        final_xs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index.coalesce()))
            x = F.dropout(x, p=0.5, training=self.training)
            final_xs.append(x)

        x = self.JK(final_xs)
        x = self.lin1(x)
        return x


class GCNII_Model(torch.nn.Module):
    def __init__(self, m, m_y, hidden=64, layers=64, alpha=0.5, theta=1.):
        super(GCNII_Model, self).__init__()
        self.lin1 = nn.Linear(m, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(layers):
            self.convs.append(GCN2Conv(channels=hidden,
                                       alpha=alpha, theta=theta, layer=i + 1))
        self.lin2 = nn.Linear(hidden, m_y)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x_0 = x
        for conv in self.convs:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(conv(x, x_0, edge_index.coalesce()))
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin2(x)
        return out


class H2GCN_Prop(MessagePassing):
    def __init__(self):
        super(H2GCN_Prop, self).__init__()

    def forward(self, h, norm_adj_1hop, norm_adj_2hop):
        h_1 = torch.sparse.mm(norm_adj_1hop, h)  # if OOM, consider using torch-sparse
        h_2 = torch.sparse.mm(norm_adj_2hop, h)
        h = torch.cat((h_1, h_2), dim=1)
        return h


class H2GCN(torch.nn.Module):
    def __init__(self, m, m_y, hidden, edge_index, dropout=0.5, act='relu'):
        super(H2GCN, self).__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(m, hidden, bias=False)
        self.act = torch.nn.ReLU() if act == 'relu' else torch.nn.Identity()
        self.H2GCN_layer = H2GCN_Prop()
        self.num_layers = 1
        self.lin_final = nn.Linear((2 ** (self.num_layers + 1) - 1) * hidden, m_y, bias=False)
        # self.lin_final = nn.Linear((self.num_layers+1)*hidden, m_y, bias=False)

        adj = to_scipy_sparse_matrix(remove_self_loops(edge_index)[0])
        adj_2hop = adj.dot(adj)
        adj_2hop = adj_2hop - sp.diags(adj_2hop.diagonal())
        adj = indicator_adj(adj)
        adj_2hop = indicator_adj(adj_2hop)
        norm_adj_1hop = get_normalized_adj(adj)
        self.norm_adj_1hop = sparse_mx_to_torch_sparse_tensor(norm_adj_1hop, 'cuda')
        norm_adj_2hop = get_normalized_adj(adj_2hop)
        self.norm_adj_2hop = sparse_mx_to_torch_sparse_tensor(norm_adj_2hop, 'cuda')

    def forward(self, x, edge_index=None):
        hidden_hs = []
        h = self.act(self.lin1(x))
        hidden_hs.append(h)
        for i in range(self.num_layers):
            h = self.H2GCN_layer(h, self.norm_adj_1hop, self.norm_adj_2hop)
            hidden_hs.append(h)
        h_final = torch.cat(hidden_hs, dim=1)
        # print(f'lin_final.size(): {self.lin_final.weight.size()}, h_final.size(): {h_final.size()}')
        h_final = F.dropout(h_final, p=self.dropout, training=self.training)
        output = self.lin_final(h_final)
        return output
