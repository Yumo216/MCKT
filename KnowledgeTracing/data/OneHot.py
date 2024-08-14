import numpy as np
from torch.utils.data.dataset import Dataset
from DKT.KnowledgeTracing.Constant import Constants as C
import torch

class OneHot(Dataset):

    def __init__(self, ques, ans):
        self.ques = ques
        self.ans = ans
        self.numofques = C.QUES

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, index):
        questions = self.ques[index]
        answers = self.ans[index]
        onehot = self.onehot(questions, answers)
        return onehot

    def onehot(self, questions, answers):
        result = torch.zeros(C.MAX_STEP, 2 * self.numofques).cuda()
        for i in range(C.MAX_STEP):
            if answers[i] > 0:
                result[i][questions[i]-1] = 1
            elif answers[i] == 0:
                result[i][questions[i] + C.QUES-1] = 1
        return result
