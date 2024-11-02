import torch.nn as nn
from KnowledgeTracing.Constant import Constants as C


class testKT(nn.Module):
    def __init__(self, emb_dim, hidden_dim, layer_dim, output_dim):
        super(testKT, self).__init__()

        self.qa_emb = nn.Embedding(2 * C.QUES + 1, emb_dim)
        self.q_emb = nn.Embedding(C.QUES + 1, emb_dim)
        self.difficulty_emb = nn.Embedding(C.QUES + 1, 1)

        self.output_dim = output_dim
        self.gru = nn.RNN(emb_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, self.output_dim)
        self.predict_linear = nn.Linear(hidden_dim, C.QUES, bias=True)

    def forward(self, qa, q):  # [bs, L, 2q]
        qa_emb = self.qa_emb(qa)
        q_emb = self.q_emb(q)

        difficulty = self.difficulty_emb(q)
        com_emb = qa_emb + difficulty * q_emb  # [bs, L, emb_dim]
        out, _ = self.gru(com_emb)  # [bs,l,d]

        logit = self.fc(out)  # [bs,l,1]
        return logit
