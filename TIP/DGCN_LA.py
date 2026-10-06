from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import random
import numpy as np

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# for missingPredNew file

# dynamic graph GRU
class gconv_RNN(nn.Module):
    def __init__(self):
        super(gconv_RNN, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nvc,nvw->nwc', (x, A))
        return x.contiguous()


class gconv_hyper(nn.Module):
    def __init__(self):
        super(gconv_hyper, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nvc,vw->nwc', (x, A))
        return x.contiguous()


class gcn(nn.Module):
    def __init__(self, dims, gdep, dropout, alpha, beta, gamma, type=None):
        super(gcn, self).__init__()
        if type == 'RNN':
            self.gconv_preA = gconv_hyper()  # for static
            self.gconv = gconv_RNN()  # for graph_dy
            self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])

        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.type_GNN = type

    def forward(self, x, adj):

        h = x
        out = [h]
        if self.type_GNN == 'RNN':
            for _ in range(self.gdep):
                h = self.alpha * x + self.beta * self.gconv_preA(h.to(x.device), adj[0]) + self.gamma * self.gconv(h.to(x.device), adj[1])  # 
                out.append(h.to(x.device))

        ho = torch.cat(out, dim=-1)

        ho = self.mlp(ho)

        return ho


class GRU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GRU, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # need to redefine
        dims_hyper = [in_dim + out_dim, out_dim]
        a, b, c, d, e = 1, 0.1, 0.2, 0.8, 0.8
        self.z = gcn(dims_hyper, a, b, c, d, e, 'RNN') 
        self.z2 = gcn(dims_hyper, a, b, c, d, e, 'RNN')

        self.r = gcn(dims_hyper, a, b, c, d, e,  'RNN')
        self.r2 = gcn(dims_hyper, a, b, c, d, e,  'RNN')

        self.g = gcn(dims_hyper, a, b, c, d, e,  'RNN')
        self.g2 = gcn(dims_hyper, a, b, c, d, e,  'RNN')
        print("a:{0}, c:{1}, d:{2}".format(a, c, d))

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, inX, inH, M, adjN, adjNT):
        concat = torch.cat((inX, inH), dim=-1).float()  # 2D
        z = self.sig(self.z(concat, adjN) + self.z2(concat, adjNT))
        r = self.sig(self.r(concat, adjN) + self.r2(concat, adjNT))
        temp = torch.cat((inX, torch.mul(r, inH)), dim=-1)
        h_hat = self.tanh(self.g(temp, adjN) + self.g2(temp, adjNT))

        outHidden = torch.mul(z, h_hat) + torch.mul(1 - z, inH)

        return outHidden


class AdpGRU(nn.Module):
    def __init__(self, in_dim, out_dim, num_id, graph_size, emb_size):
        super(AdpGRU, self).__init__()
        # PGCN->GRU
        self.in_dim = in_dim  
        self.out_dim = out_dim
        self.num_id = num_id

        self.GL = nn.Parameter(torch.FloatTensor(num_id, graph_size))  # N,D
        nn.init.kaiming_uniform_(self.GL)
        self.W1 = nn.Linear(graph_size, emb_size, bias=False)

        self.W2 = nn.Linear(in_dim + emb_size, in_dim + emb_size, bias=False)
        # GRU op ---> need to redefine
        self.GRU = GRU(in_dim, out_dim)  
        # dropout+res op
        self.drop = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(out_dim)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden, M, adjN):

        B, _, _ = x.shape

        x = F.dropout(x, p=0.1, training=self.training)  
        # calculate the dynamic graph
        graph_mid = self.W1(self.GL.to(x.device).unsqueeze(0).expand(B, -1, -1))  # B,N,D
        graph_mid = self.W2(torch.cat((x, graph_mid), dim=-1))  

        graph_dy = F.softmax(F.relu(graph_mid @ graph_mid.transpose(-2, -1)),
                                                                   dim=-1).to(x.device) 

        adj = [adjN[0].float(), graph_dy.float()]
        adjT = [adjN[1].float(), graph_dy.float()]

        gru_out = self.GRU(x, hidden, M, adj, adjT) 
        out1 = F.dropout(gru_out, p=0.1, training=self.training)
        out = self.ln(out1 + x)  

        return out


class EnBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_id, graph_size, emb_size):  # 2D,2D
        super(EnBlock, self).__init__()
        # PGCN->GRU
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.GRU1 = AdpGRU(in_dim, out_dim, num_id, graph_size, emb_size) 
        self.GRU2 = AdpGRU(out_dim, out_dim, num_id, graph_size, emb_size) 

        self.drop = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x, hidden1, hidden2, M, adjN):

        x = F.dropout(x, p=0.1, training=self.training) 
        gru_out1 = self.GRU1(x, hidden1, M, adjN)
        gru_out2 = self.GRU2(gru_out1, hidden2, M, adjN)
        out1 = F.dropout(gru_out2, p=0.1, training=self.training)  
        out = self.ln(x + out1)   

        return gru_out1, gru_out2, out
