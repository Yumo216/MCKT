import torch.nn as nn
from KnowledgeTracing.Constant import Constants as C



class DKT(nn.Module):
    def __init__(self, emb_dim, hidden_dim, layer_dim, output_dim):
        super(DKT, self).__init__()

        self.interaction_emb = nn.Embedding(2 * C.QUES + 1, emb_dim)

        self.output_dim = output_dim
        self.gru = nn.GRU(emb_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, self.output_dim)
        self.predict_linear = nn.Linear(hidden_dim, C.QUES, bias=True)

    def forward(self, x, _):  # [bs, L, 2q]

        x_e = self.interaction_emb(x)
        out, _ = self.gru(x_e)  # [bs,l,d]

        logit = self.fc(out)  # [bs,l,1]
        return logit
