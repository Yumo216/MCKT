import sys
from DKT.KnowledgeTracing.model.DKT import DKT
from DKT.KnowledgeTracing.model.DKTcont import DKTcont
from DKT.KnowledgeTracing.model.Mamba4KT import Mamba4KT
from DKT.KnowledgeTracing.model.MambaCont import MambaCont
# from DKT.KnowledgeTracing.data.dataloader import getTrainLoader, getTestLoader, getLoader
from DKT.KnowledgeTracing.data.dataloader import getDataLoader
from DKT.KnowledgeTracing.Constant import Constants as C
import json
import torch.optim as optim
from DKT.KnowledgeTracing.evaluation import eval
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

with open('../../KTDataset/XES3G5M/qid2content_emb.json', 'r', encoding='utf-8') as f:
    qid2cont = (json.load(f))  # dict [Q, 768]

ques_cont = torch.tensor([qid2cont[str(i)] for i in range(len(qid2cont))]).to(device)
# ques_cont = torch.tensor(ques_cont).repeat((2, 1)).to(device)  # [2Q, 768]



model_name = 'Mamba'

if model_name == "DKT":
    model = DKT(emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
                output_dim=C.OUTPUT).to(device)

elif model_name == "DKT-cont":
    model = DKTcont(emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
                output_dim=C.OUTPUT, ques_cont=ques_cont).to(device)

elif model_name == "Mamba":
    model = Mamba4KT(emb_dim=C.EMB_DIM, input_size=C.EMB_DIM, num_layers=C.MAM_LAYERS,
                     dropout_prob=C.DROP_PRO, d_state=C.d_state, d_conv=C.d_conv,
                     expand=C.expand).to(device)

elif model_name == "Mamba-cont":
    model = MambaCont(emb_dim=C.EMB_DIM, input_size=C.EMB_DIM, num_layers=C.MAM_LAYERS,
                     dropout_prob=C.DROP_PRO, d_state=C.d_state, d_conv=C.d_conv,
                      expand=C.expand, ques_cont=ques_cont).to(device)

else:
    raise ValueError(f"Unknown model name: {model_name}")

optimizer = optim.Adam(model.parameters(), lr=C.LR)

loss_func = eval.lossFunc(C.MAX_STEP, device).to(device)

# trainLoaders, testLoaders = getLoader(C.DATASET)
trainLoader, validationLoader, testLoader = getDataLoader(C.BATCH_SIZE, C.QUES, C.MAX_STEP)

best_auc = 0.0
best_epoch = 0
best_acc = 0.0

patience = 16  # 设置早停的耐心值
counter = 0

# model.init_params()    参数初始化，有机会可以试试，看看会不会对性能有提升
# model.init_embeddings() 尤其是题目embedding的初始化

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
            counter = 0  # 重置计数器
        else:
            counter += 1
    print('Best auc at present: %f  acc:  %f  Best epoch: %d' % (best_auc, best_acc, best_epoch))
    if counter >= patience:
        print(f"Early stopping triggered. No improvement in AUC for {patience} consecutive epochs.")
        print(f'Final Best AUC: {best_auc * 100:.2f}%  ACC: {best_acc * 100:.2f}%  Best epoch: {best_epoch}')
        break

    # if auc > best_auc:
    #     print('best checkpoint')
    #     best_auc = auc
# eval.test_epoch(model, testLoader, loss_func, device)




# for epoch in range(C.EPOCH):
#     print(f"Current model: {model_name}    Using GPU: {device_id}")
#     print('epoch: ' + str(epoch + 1))
#     model, optimizer = eval.train_epoch(model, trainLoaders, optimizer, loss_func)
#     with torch.no_grad():
#         auc, acc = eval.test_epoch(model, testLoaders, loss_func, device)
#         if best_auc < auc:
#             best_auc = auc
#             best_acc = acc
#             best_epoch = epoch + 1
#             counter = 0  # 重置计数器
#         else:
#             counter += 1
#         # print('Best auc at present: %f  acc:  %f  Best epoch: %d' % (best_auc, best_acc, best_epoch))
#         print(f'Best AUC at present: {best_auc * 100:.2f}%  ACC: {best_acc * 100:.2f}%  Best epoch: {best_epoch}')
#
#         if counter >= patience:
#             print(f"Early stopping triggered. No improvement in AUC for {patience} consecutive epochs.")
#             print(f'Final Best AUC: {best_auc * 100:.2f}%  ACC: {best_acc * 100:.2f}%  Best epoch: {best_epoch}')
#             break