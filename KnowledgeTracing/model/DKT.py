import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from DKT.KnowledgeTracing.Constant import Constants as C
from einops import rearrange, repeat, einsum


class DKT(nn.Module):
    def __init__(self, emb_dim, hidden_dim, layer_dim, output_dim):
        super(DKT, self).__init__()

        # emb = nn.Embedding(2 * C.QUES, emb_dim)
        # self.ques = emb(torch.LongTensor([i for i in range(2 * C.QUES)])).cuda()
        self.interaction_emb = nn.Embedding(2 * C.QUES + 1, emb_dim)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.gru = nn.RNN(emb_dim, hidden_dim, layer_dim, batch_first=True)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=2,
                                          batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.output_dim)
        self.predict_linear = nn.Linear(hidden_dim, C.QUES, bias=True)

    def forward(self, x, _):  # shape of input: [batch_size, length, 2q ]   [batch_size, length]

        # x_e = x.matmul(self.ques)  # x_d [64,50,emb_dim]
        x_e = self.interaction_emb(x)  #
        '''gru'''
        out, _ = self.gru(x_e)  # [bs,l,d]

        '''transformer'''
        # x_d = x_d.permute(1, 0, 2)
        #
        # mask = None
        # out = self.transformer(x_d, x_d, src_mask=mask, tgt_mask=mask)

        # Permute the output back to [batch_size, length, hidden_size]
        # out = out.permute(1, 0, 2)

        # res = self.sig(self.fc(out))
        # logit = self.fc(out)
        logit = self.fc(out)  # [bs,l,1]
        return logit
