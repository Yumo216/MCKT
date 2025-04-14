import numpy as np
import itertools
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


class DataReader():
    def __init__(self, train_path, test_path, maxstep, num_ques):
        self.train_path = train_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.num_ques = num_ques

    def getData(self, file_path):
        datas = []
        with open(file_path, 'r') as file:
            for lens, ques, skill, ans in itertools.zip_longest(*[file] * 4):
                lens = int(lens.lstrip('\ufeff').strip().strip(','))
                ques = [int(q) for q in ques.strip().strip(',').split(',')]
                ans = [int(a) for a in ans.strip().strip(',').split(',')]
                slices = lens // self.maxstep + (1 if lens % self.maxstep > 0 else 0)
                for i in range(slices):
                    data = np.zeros(shape=[self.maxstep, 3])  # 0 ->question ID
                    if lens > 0:  # 1->question ID + C.Ques or 0
                        if lens >= self.maxstep:  # 2->label (0->1, 1->2)
                            steps = self.maxstep
                        else:
                            steps = lens
                        for j in range(steps):
                            data[j][0] = ques[i * self.maxstep + j]
                            data[j][2] = ans[i * self.maxstep + j] + 1
                            if ans[i * self.maxstep + j] == 1:
                                data[j][1] = ques[i * self.maxstep + j]
                            else:
                                data[j][1] = ques[i * self.maxstep + j] + self.num_ques
                        lens = lens - self.maxstep
                    datas.append(data.tolist())
            print('done: ' + str(np.array(datas).shape))

        return datas

    # 单独划分验证集:
    def getTrainData(self):
        print('loading train data...')
        Data = np.array(self.getData(self.train_path))
        trainData, valiData = train_test_split(Data, test_size=0.2, random_state=3)
        return np.array(trainData), np.array(valiData)


    def getTestData(self):
        print('loading test data...')
        testData = self.getData(self.test_path)
        return np.array(testData)

