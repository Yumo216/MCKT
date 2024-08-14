from mamba_ssm import Mamba
import yaml
import torch
import torch.nn as nn
from DKT.KnowledgeTracing.Constant import Constants as C


class Mamba4KT(nn.Module):
    def __init__(self, emb_dim, input_size, num_layers, dropout_prob, d_state, d_conv, expand):
        super(Mamba4KT, self).__init__()

        emb = nn.Embedding(2 * C.QUES, emb_dim)
        self.ques = emb(torch.LongTensor([i for i in range(2 * C.QUES)])).cuda()


        self.interaction_emb = nn.Embedding(2 * C.QUES, emb_dim)
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # some trick before Mamba
        self.LayerNorm = nn.LayerNorm(emb_dim, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)


        # Hyperparameters for Mamba block
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.input_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])
        self.fc = nn.Linear(self.input_size, C.QUES)
        self.sig = nn.Sigmoid()

    def forward(self, x, _):  # shape of input: [batch_size, length, 2q ]

        # item_emb = x.matmul(self.ques)  # x_d [BS,L,128]
        item_emb = self.interaction_emb(x)

        # item_emb = self.dropout(item_emb)
        # item_emb = self.LayerNorm(item_emb)


        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](item_emb)

        logit = self.fc(item_emb)
        return logit


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()

        self.num_layers = num_layers
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=d_model,  # 模型维度  emb_dim
            d_state=d_state,  # 状态空间的维度  32
            d_conv=d_conv,  # 卷积核的维度  4
            expand=expand,  # 扩展因子？ 2
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor):  # [batch_size, length, emb_dim]
        hidden_states = self.mamba(input_tensor)  # [BS,L,256]
        if self.num_layers == 1:  # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:  # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        # hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
