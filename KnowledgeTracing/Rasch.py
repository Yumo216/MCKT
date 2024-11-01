import torch
import torch.nn as nn


class KnowledgeTracingGRU(nn.Module):
    def __init__(self, num_students, num_questions, emb_dim, gru_hidden_dim):
        super(KnowledgeTracingGRU, self).__init__()

        # 学生和题目的嵌入层
        self.student_emb = nn.Embedding(num_students, emb_dim)
        self.k_difficulty = nn.Embedding(num_questions, emb_dim)

        # GRU 层
        self.gru = nn.GRU(input_size=2 * emb_dim, hidden_size=gru_hidden_dim, batch_first=True)

        # 全连接层，用于最终输出概率
        self.fc = nn.Linear(gru_hidden_dim, 1)

    def forward(self, x, student_ids):  # [bs, L, 2q]
        '''
        :param x: LongTensor, 题目ID和作答情况，形状为 [batch_size, sequence_length, 1]
        :param student_ids: LongTensor, 学生ID，形状为 [batch_size]
        :return: FloatTensor, 回答正确的概率，形状为 [batch_size, sequence_length, 1]
        '''

        # 获取学生的嵌入
        stu_emb = self.student_emb(student_ids)  # [batch_size, emb_dim]
        stu_emb = stu_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # [batch_size, sequence_length, emb_dim]

        # 获取题目的难度嵌入
        ques_ids = x[:, :, 0]  # 提取出题目ID
        k_difficulty = self.k_difficulty(ques_ids)  # [batch_size, sequence_length, emb_dim]

        # 计算学生能力与题目难度之间的差值
        student_knowledge_diff = stu_emb - k_difficulty  # [batch_size, sequence_length, emb_dim]

        # 获取交互嵌入并与学生能力差值结合
        x_e = self.interaction_emb(x)  # 原始题目交互的嵌入 [batch_size, sequence_length, emb_dim]
        gru_input = torch.cat([x_e, student_knowledge_diff],
                              dim=-1)  # 合并为 GRU 输入 [batch_size, sequence_length, 2 * emb_dim]

        # GRU 层
        out, _ = self.gru(gru_input)  # [batch_size, sequence_length, gru_hidden_dim]

        # 全连接层
        logit = self.fc(out)  # [batch_size, sequence_length, 1]

        return logit

def calculate_difficulty(x):
    '''
    :param x: LongTensor, 输入形状为 [batch_size, sequence_length, 2q]
    :return: FloatTensor, 计算后的题目难度 k_difficulty
    '''
    # 假设 x[:, :, 0] 是题目 ID, x[:, :, 1] 是回答正确与否 (0 或 1)
    ques_ids = x[:, :, 0]
    responses = x[:, :, 1]
    batch_size, sequence_length = ques_ids.shape

    # 计算每个题目的回答次数和平均正确率
    difficulties = torch.zeros(batch_size, sequence_length)
    for b in range(batch_size):
        for l in range(sequence_length):
            q_id = ques_ids[b, l].item()
            response_count = (ques_ids == q_id).sum().item()
            if response_count >= 4:
                correct_responses = (responses[ques_ids == q_id] == 1).sum().item()
                difficulty = (correct_responses / response_count) * 10
                difficulties[b, l] = difficulty / 5
            else:
                difficulties[b, l] = 5
    return difficulties
