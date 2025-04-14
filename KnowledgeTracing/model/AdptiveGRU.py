import torch.nn as nn
from KnowledgeTracing.Constant import Constants as C
import torch.nn.functional as F

class DilatedResBlock(nn.Module):
    def __init__(self, dilation, channel, max_len):
        super(DilatedResBlock, self).__init__()
        self.dilation = dilation
        self.channel = channel
        self.half_channel = int(channel / 2)
        self.max_len = max_len

        self.reduce = nn.Conv1d(channel, self.half_channel, 1)
        self.masked = nn.Conv1d(self.half_channel, self.half_channel, 3, dilation=dilation)
        self.increase = nn.Conv1d(self.half_channel, channel, 1)

        self.reduce_norm = nn.LayerNorm(normalized_shape=channel)
        self.masked_norm = nn.LayerNorm(normalized_shape=self.half_channel)
        self.increase_norm = nn.LayerNorm(normalized_shape=self.half_channel)

    def forward(self, x):
        y = self.reduce_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        # y = self.reduce_norm(x)

        y = F.leaky_relu(x)
        y = self.reduce(y)

        y = self.masked_norm(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = F.leaky_relu(y)
        y = F.pad(y, pad=(2 + (self.dilation - 1) * 2, 0), mode='constant')
        y = self.masked(y)

        y = self.increase_norm(y.permute(0, 2, 1)).permute(0, 2, 1)
        # y = self.increase_norm(y)
        y = F.leaky_relu(y)
        y = self.increase(y)

        return x + y

class AdptiveGRU(nn.Module):
    """

    """

    def __init__(self, emb_dim, output_dim, layer_dim=2):

        super(AdptiveGRU, self).__init__()
        self.emb_dim = emb_dim
        self.channel = emb_dim
        self.output_dim = output_dim
        self.max_len = C.MAX_STEP

        self.dilations = [1, 2, 4, 8]
        self.hidden_layers = nn.Sequential(
            *[nn.Sequential(*[DilatedResBlock(d, emb_dim, self.max_len) for d in self.dilations]) for _ in
              range(layer_dim)])
        self.GRU = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.final_layer = nn.Linear(emb_dim, self.output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.hidden_layers(x).permute(0, 2, 1)
        x, _ = self.GRU(x)

        x = self.final_layer(x)

        return x