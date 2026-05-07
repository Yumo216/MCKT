import json
import torch
from torch import nn
from mamba_ssm import Mamba
from KnowledgeTracing.Constant import Constants as C
from .Mamba.EXBimamba import ExBimamba
from .AdptiveGRU import AdptiveGRU

json_file_path = '../../KTDataset/assist2017/AS17_diff.json'
# '../../KTDataset/XES3G5M/XES_diff.json'

class MambaCont(nn.Module):
    def __init__(self, emb_dim, input_size, num_layers, dropout_prob, d_state, ques_cont):
        super(MambaCont, self).__init__()
        with open(json_file_path, 'r') as json_file:
            self.question_difficulty = json.load(json_file)

        self.q_emb = nn.Embedding(C.QUES + 1, emb_dim)
        self.qa_emb = nn.Embedding(2 * C.QUES + 1, emb_dim)

        self.ques_cont = ques_cont

        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.summary_attn = nn.Sequential(
            nn.Linear(emb_dim, 1),  # [B,L,D] -> [B,L,1]
            nn.Softmax(dim=1)
        )

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
        self.fusion_fc = nn.Linear(emb_dim+768, emb_dim)
        self.diff_emb = nn.Linear(1, emb_dim)
        self.gru = AdptiveGRU(emb_dim=C.EMB_DIM, output_dim=C.EMB_DIM)
        self.mix = MixingLayer(emb_dim)
        self.summary_gate = nn.Linear(emb_dim, emb_dim)

        self.LayerNorm = nn.LayerNorm(emb_dim, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.last_attn = None
        self.last_q_diff = None

    def forward(self, qa, q):  # shape of input: [batch_size, length, 2q ]
        q_emb = self.q_emb(q)
        qa_emb = self.qa_emb(qa)  # [BS,L,256]
        q_cont = self.ques_cont[q]  # [BS,L,768]

        '''Early fusion'''
        item_emb = self.fusion_fc(torch.cat((qa_emb, q_cont), dim=-1))  # [BS,L,1024]

        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        '''Rasch diff'''
        difficulty = torch.tensor([self.question_difficulty.get(str(int(q_id)), 0.5) for q_id in q.view(-1)],
                                  dtype=torch.float32)
        q_diff = difficulty.view(q.size(0), q.size(1), 1).to(q_emb.device)
        # q_diff_emb = (1 - q_diff) * q_emb
        q_diff_emb = self.diff_emb(q_diff)

        '''AdptiveGRU'''
        logit_gru = self.gru(item_emb)  # [bs, L, emb_dim]

        '''Parallel Mamba'''
        for i in range(self.num_layers):
            logit_hidd, logit_diff = self.mamba_layers[i](item_emb, q_diff_emb)  # [bs, L, emb_dim]




        interaction = torch.matmul(logit_hidd, logit_diff.transpose(-1, -2))  # [B,L,L]
        interaction = torch.softmax(interaction, dim=-1)
        cross_info = torch.matmul(interaction, logit_diff)  # [B,L,D]
        logit_hidd = logit_hidd + cross_info  # 残差融合

        logit = self.mix(logit_hidd, logit_diff, logit_gru)

        # summary = torch.mean(logit, dim=1, keepdim=True)  # [B,1,D]
        attn_weights = self.summary_attn(logit)  # [B,L,1]
        summary = torch.sum(attn_weights * logit, dim=1, keepdim=True)  # [B,1,D]
        gate = torch.sigmoid(self.summary_gate(logit))  # [B,L,D]
        logit = logit + gate * summary

        res = self.fc(logit)  # [BS,L,]

        # ===== for visualization =====
        self.last_attn = attn_weights.squeeze(-1).detach()   # [B,L]
        self.last_q_diff = q_diff.squeeze(-1).detach()       # [B,L]

        return res


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
        self.weight_raw = nn.Parameter(torch.tensor([0.3, 0.1, 0.6]))

        # Optional transformation layer
        self.linear = nn.Linear(emb_dim, emb_dim, bias=True)

    def forward(self, hidd_out, diff_out, gru_out):
        # Weighted sum of the three branches
        weights = torch.softmax(self.weight_raw, dim=0)
        z0 = weights[0] * hidd_out + weights[1] * diff_out + weights[2] * gru_out

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
