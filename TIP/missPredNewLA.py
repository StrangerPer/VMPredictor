import torch
from torch import nn
from .DGCN_LA import EnBlock
from .attenion import TemporalSelfAttentionLayer
from .PGCNArch import AdapMaskEmbedding, feature_attention
from torch.autograd import Variable
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


class MissPredNew(nn.Module):
    def __init__(self, input_len, num_id, out_len, emb_size, adjN, graph_size):
        super(MissPredNew, self).__init__()

        ### basic parameter
        self.input_len = input_len
        self.out_len = out_len
        self.num_id = num_id
        self.emb_size = emb_size
        self.adjN = adjN
        self.graph_size = graph_size
        # self.Hid_size = Hid_size
        self.num_layers = 2
        print("attn:", self.num_layers)
        self.tod_embedding_dim = emb_size  
        self.dow_embedding_dim = emb_size
        self.adaptive_embedding_dim = emb_size
        self.model_dim = (
                emb_size  # 24
                + self.tod_embedding_dim
                + self.dow_embedding_dim
                + self.adaptive_embedding_dim
        )

        self.adp_layer = AdapMaskEmbedding(emb_size, input_len, num_id)
        self.enhance_layer = feature_attention(emb_size, emb_size)

        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(288, self.tod_embedding_dim)
        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding_t = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(input_len, num_id, self.adaptive_embedding_dim))  # T,N,D
            ) 

        self.G2RU = EnBlock(self.model_dim, self.model_dim, num_id, graph_size, emb_size)
        self.attnT = nn.ModuleList(
            [
                TemporalSelfAttentionLayer(self.model_dim, num_heads=4)
                for _ in range(self.num_layers)
            ]
        )

        # output layer
        self.output_proj = nn.Linear(
            input_len * (self.model_dim), out_len * 1
        )

    def forward(self, history_data):
        B, T, N, _ = history_data.shape
        batch_size = history_data.shape[0]
        x = history_data[..., :1]
        mask = history_data[..., 3:]  # B,T,N,1

        if self.tod_embedding_dim > 0:
            tod = history_data[..., 1]
        if self.dow_embedding_dim > 0:
            dow = history_data[..., 2]
        # adpEnhance layer
        x_adp = self.adp_layer(x.permute(0, 2, 1, 3), mask.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_enhance = self.enhance_layer(x_adp.permute(0, 2, 1, 3))  

        x_enhance = x_enhance.to(history_data.device)
        features = [x_enhance]

        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * 288).long()  
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim), attention--->max is 6
            features.append(tod_emb.to(history_data.device))
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                (dow * 7).long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim), attention--->max is 287
            features.append(dow_emb.to(history_data.device))

        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding_t.expand(size=(batch_size, *self.adaptive_embedding_t.shape))
            features.append(adp_emb)
        x_in = torch.cat(features, dim=-1)
        x_in = x_in.to(history_data.device)

        # first G2GRU
        ht1 = nn.init.orthogonal_(Variable(torch.zeros(B, N, self.model_dim).to(history_data.device)))
        ht2 = nn.init.orthogonal_(Variable(torch.zeros(B, N, self.model_dim).to(history_data.device)))
        end = datetime.datetime.now()
        Hid = []
        for i in range(self.input_len):
            ht1, ht2, ht = self.G2RU(x_in[:, i, :, :], ht1, ht2, mask[:, i, :, :], self.adjN) 
            Hid.append(ht.unsqueeze(-1).to(history_data.device))  # B,N,D,1

        HidAll = torch.cat(Hid, dim=-1).permute(0, 3, 1, 2)  # B,T,N,D

        for attnT in self.attnT: 
            HidAll = attnT(HidAll, dim=1)  # B,T,N,D

        HidAll = HidAll.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
        out = HidAll.reshape(
            batch_size, self.num_id, self.input_len * (self.model_dim) 
        )
        out = self.output_proj(out).view(
            batch_size, self.num_id, self.out_len, 1
        )
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)

        return out.squeeze(-1)
