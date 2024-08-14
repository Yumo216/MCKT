import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from DKT.KnowledgeTracing.Constant import Constants as C
from einops import rearrange, repeat, einsum


class DKTcont(nn.Module):
    def __init__(self, emb_dim, hidden_dim, layer_dim, output_dim, ques_cont):
        super(DKTcont, self).__init__()

        # emb = nn.Embedding(2 * C.QUES, emb_dim)
        # self.ques = emb(torch.LongTensor([i for i in range(2 * C.QUES)])).cuda()
        self.interaction_emb = nn.Embedding(2 * C.QUES + 1, emb_dim)

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.ques_cont = ques_cont
        self.gru = nn.GRU(emb_dim * 4, hidden_dim, layer_dim, batch_first=True)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=2,
                                          batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.output_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*3)

    def forward(self, x, q_id):  # shape of input: [batch_size, length, 2q ]
        x_e = self.interaction_emb(x)  # [BS,200,256]
        x_cont = self.ques_cont[q_id]  # [BS,200,768]

        # x_cont = x.matmul(self.ques_cont)  # [64,200,768]

        '''Add'''
        # x_d_1 = F.pad(x_d, (0, 768))
        # x_cont = F.pad(x_cont, (256, 0))
        # input = x_d_1 + x_cont  # [64,200,256+768]

        '''Early fusion'''
        input = torch.cat((x_e, x_cont), dim=-1)
        '''gru'''
        out, _ = self.gru(input)
        '''transformer'''
        # x_d = x_d.permute(1, 0, 2)
        #
        # mask = None
        # out = self.transformer(x_d, x_d, src_mask=mask, tgt_mask=mask)

        # Permute the output back to [batch_size, length, hidden_size]
        # out = out.permute(1, 0, 2)

        logit = self.fc(out)
        return logit
