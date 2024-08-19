import sys
sys.path.append('../')
import numpy as np
import torch.utils.data as Data
import torch
import torch.utils.data as Data
from .readdata import DataReader
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.data.readdata import DataReader
from KnowledgeTracing.data.OneHot import OneHot

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
    if dataset == 'assist2009':
        trainPath = C.Dpath + '/assist2009/assist2009_pid_train.csv'
        testPath = C.Dpath + '/assist2009/assist2009_pid_test.csv'

    elif dataset == 'assist2012':
        trainPath = C.Dpath + '/assist2012/assist2012_pid_train.csv'
        testPath = C.Dpath + '/assist2012/assist2012_pid_test.csv'

    elif dataset == 'assist2017':
        trainPath = C.Dpath + '/assist2017/assist2017_pid_train.csv'
        testPath = C.Dpath + '/assist2017/assist2017_pid_test.csv'

    elif dataset == 'ednet':
        trainPath = C.Dpath + '/ednet/ednet_pid_train.csv'
        testPath = C.Dpath + '/ednet/ednet_pid_test.csv'

    elif dataset == 'XES3G5M':
        trainPath = C.Dpath + '/XES3G5M/original_train.csv'
        testPath = C.Dpath + '/XES3G5M/original_test.csv'

    else:
        raise ValueError(f"Dataset {dataset} is not recognized.")

    return trainPath, testPath

# def getTrainLoader(train_data_path):
#     handle = DataReader(train_data_path ,C.MAX_STEP, C.QUES)
#     trainques, trainans = handle.getTrainData()
#     # trainques, trainans = np.load('../data/trainXES.npz').values()
#     dtrain = OneHot(trainques, trainans)
#     trainLoader = Data.DataLoader(dtrain, batch_size=C.BATCH_SIZE, shuffle=True)
#     return trainLoader
#
# def getTestLoader(test_data_path):
#     handle = DataReader(test_data_path, C.MAX_STEP, C.QUES)
#     testques, testans = handle.getTestData()
#     # testques, testans = np.load('../data/testXES.npz').values()
#     dtest = OneHot(testques, testans)
#     testLoader = Data.DataLoader(dtest, batch_size=C.BATCH_SIZE, shuffle=False)
#     return testLoader
#
# def getLoader(dataset):
#     trainLoaders = []
#     testLoaders = []
#     if dataset == 'assist2009':
#         trainLoader = getTrainLoader(C.Dpath + '/assist2009/assist2009_pid_train.csv')
#         trainLoaders.append(trainLoader)
#         testLoader = getTestLoader(C.Dpath + '/assist2009/assist2009_pid_test.csv')
#         testLoaders.append(testLoader)
#     elif dataset == 'assist2012':
#         trainLoader = getTrainLoader(C.Dpath + '/assist2012/assist2012_pid_train.csv')
#         trainLoaders.append(trainLoader)
#         testLoader = getTestLoader(C.Dpath + '/assist2012/assist2012_pid_test.csv')
#         testLoaders.append(testLoader)
#     elif dataset == 'assist2017':
#         trainLoader = getTrainLoader(C.Dpath + '/assist2017/assist2017_pid_train.csv')
#         trainLoaders.append(trainLoader)
#         testLoader = getTestLoader(C.Dpath + '/assist2017/assist2017_pid_test.csv')
#         testLoaders.append(testLoader)
#     elif dataset == 'ednet':
#         trainLoader = getTrainLoader(C.Dpath + '/ednet/ednet_pid_train.csv')
#         trainLoaders.append(trainLoader)
#         testLoader = getTestLoader(C.Dpath + '/ednet/ednet_pid_test.csv')
#         testLoaders.append(testLoader)
#     elif dataset == 'XES3G5M':
#         trainLoader = getTrainLoader(C.Dpath + '/XES3G5M/original_train.csv')
#         trainLoaders.append(trainLoader)
#         testLoader = getTestLoader(C.Dpath + '/XES3G5M/original_test.csv')
#         testLoaders.append(testLoader)
#
#     return trainLoaders[0], testLoaders[0]


