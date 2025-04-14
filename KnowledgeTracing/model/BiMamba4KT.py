import torch
import torch.nn as nn
import json
from KnowledgeTracing.Constant import Constants as C
from .Mamba.EXBimamba import ExBimamba
from .AdptiveGRU import AdptiveGRU

json_file_path = '../../KTDataset/assist2017/AS17_diff.json'


class BiMamba4KT(nn.Module):
    def __init__(self, emb_dim, input_size, num_layers, dropout_prob, d_state):
        super(BiMamba4KT, self).__init__()
        with open(json_file_path, 'r') as json_file:
            self.question_difficulty = json.load(json_file)

        self.q_emb = nn.Embedding(C.QUES + 1, emb_dim)
        self.qa_emb = nn.Embedding(2 * C.QUES, emb_dim)
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(emb_dim, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        # BiMamba layers
        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.input_size,
                d_state=d_state,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])
        self.fc = nn.Linear(self.input_size, C.QUES)

        self.gru = AdptiveGRU(emb_dim=C.EMB_DIM, output_dim=C.EMB_DIM)
        self.mix = MixingLayer(emb_dim)

    def forward(self, qa, q):  # shape of input: [batch_size, length]
        q_emb = self.q_emb(q)
        qa_emb = self.qa_emb(qa)
        # qa_emb = self.dropout(qa_emb)
        # qa_emb = self.LayerNorm(qa_emb)

        '''Rasch diff'''
        difficulty = torch.tensor([self.question_difficulty.get(str(int(q_id)), 0.5) for q_id in q.view(-1)],
                                  dtype=torch.float32)
        q_diff = difficulty.view(q.size(0), q.size(1), 1).to(q_emb.device)
        q_diff_emb = (1 - q_diff) * q_emb

        '''AdptiveGRU'''
        logit_gru = self.gru(qa_emb)  # [bs, L, emb_dim]

        '''Parallel Mamba'''
        for i in range(self.num_layers):
            logit_hidd, logit_diff = self.mamba_layers[i](qa_emb, q_diff_emb)  # [bs, L, emb_dim]

        logit = self.mix(logit_hidd, logit_diff, logit_gru)

        logit = self.fc(logit)  # [BS,L,emb_dim(256)-q]

        return logit


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers

        # Use BiMamba
        self.bimamba = ExBimamba(d_model=d_model, d_state=d_state)

        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor, q_diff_emb):
        # Use BiMambaEncoder for modeling
        hidd_out, diff_out = self.bimamba(input_tensor, q_diff_emb)
        if self.num_layers == 1:  # one layer without residual connection
            hidd_out = self.LayerNorm(self.dropout(hidd_out))
            diff_out = self.LayerNorm(self.dropout(diff_out))
        else:  # stacked layers with residual connections
            hidd_out = self.LayerNorm(self.dropout(hidd_out) + input_tensor)
            diff_out = self.LayerNorm(self.dropout(diff_out) + q_diff_emb)
        hidd_out = self.ffn(hidd_out)
        diff_out = self.ffn(diff_out)
        return hidd_out, diff_out


class MixingLayer(nn.Module):
    def __init__(self, emb_dim):
        super(MixingLayer, self).__init__()
        # Learnable weights for fusion
        self.a1 = nn.Parameter(torch.tensor(0.33))  # for mamba_hidden
        self.a2 = nn.Parameter(torch.tensor(0.33))  # for mamba_diff
        self.a3 = nn.Parameter(torch.tensor(0.33))  # for adaptive_gru

        # Optional transformation layer
        self.linear = nn.Linear(emb_dim, emb_dim, bias=True)

    def forward(self, hidd_out, diff_out, gru_out):
        """
        Inputs:
            hidd_out: [batch_size, L, D] - from Mamba (response stream)
            diff_out: [batch_size, L, D] - from Mamba (difficulty stream)
            gru_out: [batch_size, L, D] - from AdaptiveGRU
        Returns:
            fused representation: [batch_size, L, D]
        """
        # Weighted sum of the three branches
        z0 = self.a1 * hidd_out + self.a2 * diff_out + self.a3 * gru_out  # [B, L, D]

        # Linear transformation
        z_hat = self.linear(z0)  # [B, L, D]
        return z_hat


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
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
