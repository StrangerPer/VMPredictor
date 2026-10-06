import torch.nn as nn
import torch
import random
import numpy as np

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class feature_attention(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=5, rate=4): 
        super(feature_attention, self).__init__()
        print("kernel:{}, rate:{}".format(kernel_size, rate))
        hidden_dim = max(1, int(output_dim / rate))
        self.nconv = nn.Conv2d(input_dim, output_dim, kernel_size=(1, 1))
        self.channel_attention = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(output_dim, hidden_dim, kernel_size=(1, kernel_size),
                      padding=(0, (kernel_size - 1) // 2)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=(1, kernel_size),
                      padding=(0, (kernel_size - 1) // 2)),
            nn.BatchNorm2d(output_dim)
        )

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.nconv(x)
        x_permute = x.permute(0, 2, 3, 1)

        x_att_permute = self.channel_attention(x_permute)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * (1 + x_channel_att)

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out.permute(0, 2, 3, 1)


class AdapMaskEmbedding(nn.Module):
    def __init__(self, x_dim, imp_len, num_nodes):
        super(AdapMaskEmbedding, self).__init__()
        self.in_dim = x_dim
        self.imp_len = imp_len
        self.num_nodes = num_nodes

        self.adpMaskEmb = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(self.num_nodes, self.imp_len, self.in_dim))
        )
        self.FC1 = nn.Linear(1, self.in_dim) 

    def forward(self, x, mask):
        # x, mask: [B, N, T, C], [B, N, T, 1]
        B = x.shape[0]
        adpMask = self.adpMaskEmb.expand(size=(B, *self.adpMaskEmb.shape)).to(x.device)
        x = self.FC1(x)
        out = x * mask + (1 - mask) * adpMask
        return out
