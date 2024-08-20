import tqdm
import torch
from KnowledgeTracing.Constant import Constants as C
import torch.nn as nn
from sklearn import metrics


def performance(ground_truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().cpu().numpy(),
                                             prediction.detach().cpu().numpy())
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(ground_truth.detach().cpu().numpy(), torch.round(prediction).detach().cpu().numpy())
    print('auc: ' + str(auc) + ' acc: ' + str(acc))

    return auc, acc


class lossFunc(nn.Module):
    def __init__(self, max_step, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.q = C.QUES
        self.sig = nn.Sigmoid()
        self.max_step = max_step
        self.device = device

    def forward(self, pred, datas):  # pred [BS, L, q]
        qshft = datas[0] - 1
        qshft_adjusted = torch.where(qshft < 0, torch.tensor(0, device=self.device),
                                     qshft)
        pred_first_49 = pred[:, :C.MAX_STEP - 1, :]  # [BS, L-1, C.Ques]

        qshft_last_49 = qshft_adjusted[:, 1:C.MAX_STEP, :]  # [BS, L-1, 1]
        pred_one = torch.gather(pred_first_49, dim=-1, index=(qshft_last_49).long())

        target = datas[2]
        target = target[:, 1:, :]

        target_1d = target.reshape(-1, 1)  # [bs*(L-1), 1]
        mask = target_1d.ge(1)  # [bs*(L-1), 1]
        pred_1d = pred_one.reshape(-1, 1)  # [bs*(L-1), 1]

        filtered_pred = torch.masked_select(pred_1d, mask)  # [bs*(L-1) - be masked, 1]
        filtered_target = torch.masked_select(target_1d, mask) - 1
        pred = self.sig(filtered_pred)

        loss = self.crossEntropy(pred, filtered_target.float())

        return loss, pred, filtered_target.float()


def train_epoch(model, trainLoader, optimizer, loss_func, device):
    # global loss
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch = batch.to(device)
        # shape of a batch:[bs, L, 2 * ques]
        datas = torch.chunk(batch, 3, 2)
        logit = model(datas[1].squeeze(-1), datas[0].squeeze(-1))
        loss, p, a = loss_func(logit, datas)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    return model, optimizer


def test_epoch(model, testLoader, loss_func, device):
    ground_truth = torch.tensor([], device=device)
    prediction = torch.tensor([], device=device)
    loss = torch.tensor([], device=device)
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        batch = batch.to(device)
        datas = torch.chunk(batch, 3, 2)
        logit = model(datas[1].squeeze(-1), datas[0].squeeze(-1))
        loss, p, a = loss_func(logit, datas)
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, a])
    print('loss:', loss.item())
    return performance(ground_truth, prediction)
