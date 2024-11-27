import torch
import torch.nn as nn
import json
from mamba_ssm import Mamba
from KnowledgeTracing.Constant import Constants as C

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

    def forward(self, qa, q):  # shape of input: [batch_size, length, 2q]
        q_emb = self.q_emb(q)



        qa_emb = self.qa_emb(qa)
        qa_emb = self.dropout(qa_emb)
        qa_emb = self.LayerNorm(qa_emb)

        # 获取每个题目的难度
        difficulty = torch.tensor([self.question_difficulty.get(str(int(q_id)), 0.5) for q_id in q.view(-1)],
                                  dtype=torch.float32)
        q_diff = difficulty.view(q.size(0), q.size(1), 1).to(q_emb.device)  # 调整维度与 q_emb 匹配
        q_diff_emb = (1-q_diff) * q_emb

        # 结合 Rasch 模型：qa 作为学生能力，1-difficulty 作为题目难度
        # com_emb = qa_emb - (1-q_diff_emb) * q_emb  # [bs, L, emb_dim]

        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](qa_emb, q_diff_emb)

        logit = self.fc(item_emb)
        return logit


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers

        # Use BiMambaEncoder
        self.bimamba = BiMambaEncoder(d_model=d_model, d_state=d_state)

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


class BiMambaEncoder(nn.Module):
    def __init__(self, d_model, d_state):
        super(BiMambaEncoder, self).__init__()
        self.d_model = d_model

        # Forward and backward Mamba modules
        self.mamba = Mamba(d_model, d_state)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, diff):
        # Residual connection of the original input
        residual1 = x


        '''
        residual2 = diff
        x_flip = torch.flip(diff, dims=[1])  # Flip Sequence
        mamba_out_backward = self.mamba(x_flip)
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back
        '''
        # Forward Mamba
        x_norm = self.norm1(x)
        mamba_out_forward = self.mamba(x_norm)

        # # Backward Mamba
        x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        # x_flip = x_norm
        mamba_out_backward = self.mamba(x_flip)
        '''mask'''
        # 创建下三角掩码
        seq_len = x.size(1)
        device = x.device

        simple_mask = torch.tril(torch.ones(seq_len, 1, device=device)).unsqueeze(0)  # (1, seq_len, 1)
        mamba_out_backward = mamba_out_backward * simple_mask  # 逐元素相乘

        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back


        # Combining forward and backward
        mamba_out = mamba_out_forward + mamba_out_backward

        mamba_out = self.norm2(mamba_out)
        ff_out = self.feed_forward(mamba_out)

        output = ff_out + residual1
        return output

    @staticmethod
    def generate_future_mask(seq_len):
        """
        Generate a mask to prevent access to future time steps.
        Args:
            seq_len (int): Length of the input sequence.
        Returns:
            torch.Tensor: A (seq_len, seq_len) mask tensor.
        """
        # Upper triangular matrix with 1s in the upper triangle
        mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
        # Convert 1s to -inf for masking
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    @staticmethod
    def apply_mask(input_tensor, mask):
        """
        Apply a mask to the input tensor to prevent access to future information.
        Args:
            input_tensor (torch.Tensor): Shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Shape (seq_len, seq_len).
        Returns:
            torch.Tensor: Masked input tensor.
        """
        batch_size, seq_len, d_model = input_tensor.size()
        # Expand mask to match batch size and model dimension
        expanded_mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        masked_tensor = torch.bmm(expanded_mask, input_tensor)  # (batch_size, seq_len, d_model)

        return masked_tensor


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
