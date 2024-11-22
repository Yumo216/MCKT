import sys
from KnowledgeTracing.model.DKT import DKT
from KnowledgeTracing.model.SAKT import SAKT
from KnowledgeTracing.model.storm import testKT
from KnowledgeTracing.model.DKTcont import DKTcont
from KnowledgeTracing.model.Mamba4KT import Mamba4KT
from KnowledgeTracing.model.BiMamba4KT import BiMamba4KT
from KnowledgeTracing.model.MambaCont import MambaCont
from KnowledgeTracing.data.dataloader import getDataLoader
from KnowledgeTracing.Constant import Constants as C
import json
import torch.optim as optim
from KnowledgeTracing.evaluation import eval
import torch
import numpy as np

device_id = 1
torch.cuda.set_device(device_id)
sys.path.append('../')

device = torch.device('cuda')

print('Dataset: ' + C.DATASET + ', Learning Rate: ' + str(C.LR) + '\n')

'''set random seed'''
SEED = 2024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

with open('../../KTDataset/XES3G5M/cid2content_emb.json', 'r', encoding='utf-8') as f:
    qid2cont = (json.load(f))  # dict [Q, 768]

ques_cont = torch.tensor([qid2cont[str(i)] for i in range(len(qid2cont))]).to(device)

model_name = 'BiMamba'

if model_name == "DKT":
    model = DKT(emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
                output_dim=C.OUTPUT).to(device)

elif model_name == "testKT":
    model = testKT(emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
                    output_dim=C.OUTPUT).to(device)

elif model_name == "SAKT":
    model = SAKT(emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
                 output_dim=C.OUTPUT).to(device)

elif model_name == "DKT-cont":
    model = DKTcont(emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
                    output_dim=C.OUTPUT, ques_cont=ques_cont).to(device)

elif model_name == "Mamba":
    model = Mamba4KT(emb_dim=C.EMB_DIM, input_size=C.EMB_DIM, num_layers=C.MAM_LAYERS,
                     dropout_prob=C.DROP_PRO, d_state=C.d_state, d_conv=C.d_conv,
                     expand=C.expand).to(device)

elif model_name == "BiMamba":
    model = BiMamba4KT(emb_dim=C.EMB_DIM, input_size=C.EMB_DIM, num_layers=C.MAM_LAYERS,
                       dropout_prob=C.DROP_PRO, d_state=C.d_state,).to(device)

elif model_name == "Mamba-cont":
    model = MambaCont(emb_dim=C.EMB_DIM, input_size=C.EMB_DIM, num_layers=C.MAM_LAYERS,
                      dropout_prob=C.DROP_PRO, d_state=C.d_state, d_conv=C.d_conv,
                      expand=C.expand, ques_cont=ques_cont).to(device)

else:
    raise ValueError(f"Unknown model name: {model_name}")

optimizer = optim.Adam(model.parameters(), lr=C.LR)

loss_func = eval.lossFunc(C.MAX_STEP, device).to(device)


trainLoader, validationLoader, testLoader = getDataLoader(C.BATCH_SIZE, C.QUES, C.MAX_STEP)

best_auc = 0.0
best_epoch = 0
best_acc = 0.0

patience = 16
counter = 0

optimizer = optim.Adam(model.parameters(), lr=C.LR)
best_auc = 0
for epoch in range(C.EPOCH):
    print(f"Current model: {model_name}    Using GPU: {device_id}")
    print('epoch: ' + str(epoch + 1))
    model, optimizer = eval.train_epoch(model, trainLoader, optimizer, loss_func, device)
    with torch.no_grad():
        auc, acc = eval.test_epoch(model, validationLoader, loss_func, device)
        if best_auc < auc:
            best_auc = auc
            best_acc = acc
            best_epoch = epoch + 1
            counter = 0
        else:
            counter += 1
    print('Best auc at present: %f  acc:  %f  Best epoch: %d' % (best_auc, best_acc, best_epoch))
    if counter >= patience:
        print(f"Early stopping triggered. No improvement in AUC for {patience} consecutive epochs.")
        print(f'Final Best AUC: {best_auc * 100:.2f}%  ACC: {best_acc * 100:.2f}%  Best epoch: {best_epoch}')
        break
