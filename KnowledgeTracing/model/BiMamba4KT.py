import torch
import torch.nn as nn
import json
from mamba_ssm import Mamba
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

        # AdptiveGRU
        self.gru = AdptiveGRU(emb_dim=C.EMB_DIM, output_dim=C.EMB_DIM)
        self.mix = MixingLayer(emb_dim)

    def forward(self, qa, q):  # shape of input: [batch_size, length]
        q_emb = self.q_emb(q)

        qa_emb = self.qa_emb(qa)
        qa_emb = self.dropout(qa_emb)
        qa_emb = self.LayerNorm(qa_emb)

        # 获取每个题目的难度
        difficulty = torch.tensor([self.question_difficulty.get(str(int(q_id)), 0.5) for q_id in q.view(-1)],
                                  dtype=torch.float32)
        q_diff = difficulty.view(q.size(0), q.size(1), 1).to(q_emb.device)  # 调整维度与 q_emb 匹配
        q_diff_emb = (1 - q_diff) * q_emb

        # 结合 Rasch 模型：qa 作为学生能力，1-difficulty 作为题目难度
        # com_emb = qa_emb - (1-q_diff_emb) * q_emb  # [bs, L, emb_dim]
        # item_emb = torch.zeros_like(qa_emb)
        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](qa_emb, q_diff_emb)  # [bs, L, emb_dim]

        # logit_gru = self.gru(qa_emb)  # [bs, L, emb_dim]
        # logit = self.mix(item_emb)

        logit = self.fc(item_emb)

        return logit


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers

        # Use BiMamba
        # self.bimamba = BiMambaEncoder(d_model=d_model, d_state=d_state)
        self.bimamba = ExBimamba(d_model=d_model, d_state=d_state)

        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor, q_diff_emb):
        # Use BiMambaEncoder for modeling
        hidden_states = self.bimamba(input_tensor, q_diff_emb)
        if self.num_layers == 1:  # one layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:  # stacked layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class MixingLayer(nn.Module):
    def __init__(self, emb_dim):
        super(MixingLayer, self).__init__()
        # 定义可学习参数 a1 和 a2
        self.a1 = nn.Parameter(torch.tensor(0.5))  # 初始化为 0.5，可学习
        self.a2 = nn.Parameter(torch.tensor(0.5))  # 初始化为 0.5，可学习
        # 定义线性变换层 W 和 b
        self.linear = nn.Linear(emb_dim, emb_dim, bias=True)

    def forward(self, m_output, g_output):
        """
        :param m_output: [batch_size, L, D] 来自 Mamba 的输出
        :param f_output: [batch_size, L, D] 来自 GRU 的输出
        :return: [batch_size, L, D] 融合后的表示
        """
        # 加权求和
        z0 = self.a1 * m_output + self.a2 * g_output  # [batch_size, L, D]
        # 线性变换
        z_hat = self.linear(z0)  # [batch_size, L, D]
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
