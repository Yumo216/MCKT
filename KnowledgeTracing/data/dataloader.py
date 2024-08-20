import sys

sys.path.append('../')
import numpy as np
import torch.utils.data as Data
import torch
import torch.utils.data as Data
from .readdata import DataReader
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.data.readdata import DataReader


def getDataLoader(batch_size, num_of_questions, max_step):
    train_path, test_path = getDatasetPaths(C.DATASET)
    handle = DataReader(train_path, test_path, max_step,
                        num_of_questions)
    train, vali = handle.getTrainData()
    dtrain = torch.tensor(train.astype(int).tolist(), dtype=torch.long)
    dvali = torch.tensor(vali.astype(int).tolist(), dtype=torch.long)
    dtest = torch.tensor(handle.getTestData().astype(int).tolist(),
                         dtype=torch.long)
    trainLoader = Data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    valiLoader = Data.DataLoader(dvali, batch_size=batch_size, shuffle=True)
    testLoader = Data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
    return trainLoader, valiLoader, testLoader


def getDatasetPaths(dataset):
    if dataset == 'assist2017':
        trainPath = C.Dpath + '/assist2017/assist2017_pid_train.csv'
        testPath = C.Dpath + '/assist2017/assist2017_pid_test.csv'

    elif dataset == 'XES3G5M':
        trainPath = C.Dpath + '/XES3G5M/original_train.csv'
        testPath = C.Dpath + '/XES3G5M/original_test.csv'

    else:
        raise ValueError(f"Dataset {dataset} is not recognized.")

    return trainPath, testPath
