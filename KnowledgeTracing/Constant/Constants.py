Dpath = '../../KTDataset'

datasets = {
    '2017': 'assist2017',
    'XES': 'XES3G5M',
}

question = {
    'assist2017': 3162,
    'XES3G5M': 7652,
}
skill = {
    'assist2017': 102,
    'XES3G5M': 865,
}

DATASET = datasets['2017']
QUES = question[DATASET]

# Dataloader
BATCH_SIZE = 128
MAX_STEP = 200
EMB_DIM = 256

# RNN
INPUT = QUES * 2
HIDDEN = 256
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
