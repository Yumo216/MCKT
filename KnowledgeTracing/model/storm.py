import torch.nn as nn
import json
import torch
from KnowledgeTracing.Constant import Constants as C

json_file_path = '../../KTDataset/assist2017/AS17_diff.json'


class testKT(nn.Module):
    def __init__(self, emb_dim, hidden_dim, layer_dim, output_dim):
        super(testKT, self).__init__()
        with open(json_file_path, 'r') as json_file:
            self.question_difficulty = json.load(json_file)

        self.qa_emb = nn.Embedding(2 * C.QUES + 1, emb_dim)
        self.q_emb = nn.Embedding(C.QUES + 1, emb_dim)


        self.output_dim = output_dim
        self.gru = nn.RNN(emb_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, self.output_dim)
        self.predict_linear = nn.Linear(hidden_dim, C.QUES, bias=True)

    def forward(self, qa, q):  # [bs, L, 2q]
        qa_emb = self.qa_emb(qa)
        q_emb = self.q_emb(q)

        # 获取每个题目的难度
        difficulty = torch.tensor([self.question_difficulty.get(str(int(q_id)), 0.5) for q_id in q.view(-1)],
                                  dtype=torch.float32)
        difficulty = difficulty.view(q.size(0), q.size(1), 1).to(q_emb.device)  # 调整维度与 q_emb 匹配

        # 结合 Rasch 模型：qa 作为学生能力，1-difficulty 作为题目难度
        com_emb = qa_emb - (1-difficulty) * q_emb  # [bs, L, emb_dim]
        out, _ = self.gru(com_emb)  # [bs,l,d]

        logit = self.fc(out)  # [bs,l,1]
        return logit
