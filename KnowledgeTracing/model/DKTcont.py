import torch
import torch.nn as nn
from KnowledgeTracing.Constant import Constants as C


class DKTcont(nn.Module):
    def __init__(self, emb_dim, hidden_dim, layer_dim, output_dim, ques_cont):
        super(DKTcont, self).__init__()

        self.interaction_emb = nn.Embedding(2 * C.QUES + 1, emb_dim)

        self.output_dim = output_dim
        self.ques_cont = ques_cont
        self.gru = nn.GRU(emb_dim * 4, hidden_dim, layer_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x, q_id):
        x_e = self.interaction_emb(x)
        x_cont = self.ques_cont[q_id]

        '''Add'''
        # x_d_1 = F.pad(x_d, (0, 768))
        # x_cont = F.pad(x_cont, (256, 0))
        # input = x_d_1 + x_cont  # [64,200,256+768]

        '''Early fusion'''
        input = torch.cat((x_e, x_cont), dim=-1)

        '''gru'''
        out, _ = self.gru(input)

        logit = self.fc(out)
        return logit
