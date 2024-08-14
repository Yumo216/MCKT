import numpy as np
import itertools
import tqdm
from DKT.KnowledgeTracing.Constant import Constants as C

class DataReader:
    def __init__(self, path, maxstep, numofques):
        self.path = path
        self.maxstep = maxstep
        self.numofques = numofques

    def getTrainData(self):
        trainqus = []
        trainans = []
        with open(self.path, 'r', encoding='UTF-8-sig') as train:
            for len, ques, skill, ans in tqdm.tqdm(itertools.zip_longest(*[train] * 4), desc='loading train data:    ',
                                                   mininterval=2):
                len = int(len.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)
                mod = 0 if len % self.maxstep == 0 else (self.maxstep - len % self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                trainqus = np.append(trainqus, ques).astype(int)
                trainans = np.append(trainans, ans).astype(int)
                trainqus = trainqus.reshape([-1, self.maxstep])
                trainans = trainans.reshape([-1, self.maxstep])

        # Save preprocessed training data to a .npz file
        np.savez_compressed('trainXES.npz', trainqus=trainqus, trainans=trainans)
        print("Training data has been preprocessed and saved.")
        return trainqus, trainans

    def getTestData(self):
        testqus = []
        testans = []
        with open(self.path, 'r', encoding='UTF-8-sig') as test:
            for len, ques, skill, ans in tqdm.tqdm(itertools.zip_longest(*[test] * 4), desc='loading test data:    ',
                                                   mininterval=2):
                len = int(len.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)
                mod = 0 if len % self.maxstep == 0 else (self.maxstep - len % self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                testqus = np.append(testqus, ques).astype(int)
                testans = np.append(testans, ans).astype(int)
                testqus = testqus.reshape([-1, self.maxstep])
                testans = testans.reshape([-1, self.maxstep])

        # Save preprocessed test data to a .npz file
        np.savez_compressed('testXES.npz', testqus=testqus, testans=testans)
        print("Test data has been preprocessed and saved.")
        return testqus, testans


def main():
    # 设置参数值
    path_train = '../../KTDataset/XES3G5M/original_train.csv'
    path_test = '../../KTDataset/XES3G5M/original_test.csv'
    maxstep = 200
    numofques = C.QUES

    # 处理训练数据
    reader = DataReader(path_train, maxstep, numofques)
    reader.getTrainData()

    # 处理测试数据
    reader = DataReader(path_test, maxstep, numofques)
    reader.getTestData()


if __name__ == "__main__":
    main()
