import sys
import logging
from datetime import datetime
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


device_id = 0
torch.cuda.set_device(device_id)
sys.path.append('../')

device = torch.device('cuda')

print('Dataset: ' + C.DATASET + ', Learning Rate: ' + str(C.LR) + '\n')

'''set random seed'''
SEED = 2025
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

timestamp = datetime.now().strftime("%Y%m%d_%H")


with open('../../KTDataset/XES3G5M/qid2content_emb.json', 'r', encoding='utf-8') as f:
    qid2cont = (json.load(f))  # dict [Q, 768]

ques_cont = torch.tensor([qid2cont[str(i)] for i in range(len(qid2cont))]).to(device)

model_name = 'BiMamba-cont'  # "BiMamba" for 2017, "BiMamba-cont" for

if model_name == "Mamba":
    model = Mamba4KT(emb_dim=C.EMB_DIM, input_size=C.EMB_DIM, num_layers=C.MAM_LAYERS,
                     dropout_prob=C.DROP_PRO, d_state=C.d_state, d_conv=C.d_conv,
                     expand=C.expand).to(device)

elif model_name == "BiMamba":
    model = BiMamba4KT(emb_dim=C.EMB_DIM, input_size=C.EMB_DIM, num_layers=C.MAM_LAYERS,
                       dropout_prob=C.DROP_PRO, d_state=C.d_state).to(device)

elif model_name == "BiMamba-cont":
    model = MambaCont(emb_dim=C.EMB_DIM, input_size=C.EMB_DIM, num_layers=C.MAM_LAYERS,
                      dropout_prob=C.DROP_PRO, d_state=C.d_state, ques_cont=ques_cont).to(device)

else:
    raise ValueError(f"Unknown model name: {model_name}")

save_path = f'../saved/{C.DATASET}_{model_name}_{timestamp}.pth'
log_path = f'../saved/{C.DATASET}_{model_name}_{timestamp}.log'
optimizer = optim.Adam(model.parameters(), lr=C.LR)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

#   ../saved/XES3G5M_BiMamba-cont_20250417_20.pth  ../saved/ASSIST17_BiMamba_20250623_17.pth
model.load_state_dict(torch.load('../saved/XES3G5M_BiMamba-cont_20250417_20.pth'), strict=False)


loss_func = eval.lossFunc(C.MAX_STEP, device).to(device)

trainLoader, validationLoader, testLoader = getDataLoader(C.BATCH_SIZE, C.QUES, C.MAX_STEP)

best_auc = 0.0
best_epoch = 0
best_acc = 0.0

patience = 15
counter = 0

optimizer = optim.Adam(model.parameters(), lr=C.LR)
best_auc = 0

for epoch in range(C.EPOCH):
    logging.info(f"Current model: {model_name}    Using GPU: {device_id}")
    logging.info(f"Epoch: {epoch + 1}/{C.EPOCH}")
    model, optimizer = eval.train_epoch(model, trainLoader, optimizer, loss_func, device)
    with torch.no_grad():
        auc, acc = eval.test_epoch(model, validationLoader, loss_func, device)
        if best_auc < auc:
            best_auc = auc
            best_acc = acc
            best_epoch = epoch + 1
            counter = 0
            # save best model
            # torch.save(model.state_dict(), save_path)
            logging.info(f"New best AUC: {best_auc:.4f}, ACC: {best_acc:.4f} at epoch {best_epoch} (model saved).")
        else:
            counter += 1
            logging.info(f"No improvement in AUC. Counter: {counter}/{patience}")

        if counter >= patience:
            logging.info(f"Early stopping triggered. No improvement for {patience} consecutive epochs.")
            logging.info(f"Final Best AUC: {best_auc * 100:.2f}%  ACC: {best_acc * 100:.2f}%  Best epoch: {best_epoch}")
            break

    logging.info("Training finished. Evaluating on test set using the best model...")
    logging.info(f"Loaded model from: {save_path}")

model.eval()

model.y_r_list = []
model.y_d_list = []

with torch.no_grad():
    test_auc, test_acc = eval.test_epoch(model, testLoader, loss_func, device)
    logging.info(f"[Test Result] AUC: {test_auc * 100:.2f}%  ACC: {test_acc * 100:.2f}%")


