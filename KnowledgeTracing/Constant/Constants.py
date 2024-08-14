Dpath = '../../KTDataset'

datasets = {
    '2009' : 'assist2009',
    '2017' : 'assist2017',
    'XES' : 'XES3G5M',
    'ednet': 'ednet'
}


question = {
    'assist2009' : 16891,
    'assist2017' : 3162,
    'XES3G5M' : 7652,
    'ednet': 10795
}
skill = {
    'assist2009' : 110,
    'assist2017' : 102,
    'XES3G5M' : 865,    # 待定
    'ednet': 1676,
}

DATASET = datasets['2009']
QUES = question[DATASET]


# Dataloader
BATCH_SIZE = 128
MAX_STEP = 200
EMB_DIM = 256

# RNN
INPUT = QUES * 2
HIDDEN = 128
RNN_LAYERS = 1
OUTPUT = QUES

# Mamba layer
MAM_LAYERS = 2
DROP_PRO = 0.2

# Mamba block
d_state = 32
d_conv = 4
expand = 2

# Training
LR = 0.001
EPOCH = 50


