import tqdm
import torch
from DKT.KnowledgeTracing.Constant import Constants as C
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

    def forward(self, pred, datas):   # pred [BS, L, q]
        qshft = datas[0] - 1  # [BS,L,1] 0 ~ C.Ques

        qshft_adjusted = torch.where(qshft < 0, torch.tensor(0, device=self.device),
                                            qshft)
        # 取出 pred 的前 49 个步骤
        pred_first_49 = pred[:, :C.MAX_STEP - 1, :]  # [BS, 49, C.Ques]

        # 取出 qshft_adjusted 的后 49 个元素
        qshft_last_49 = qshft_adjusted[:, 1:C.MAX_STEP, :]  # [BS, 199, 1]
        pred_one = torch.gather(pred_first_49, dim=-1, index=(qshft_last_49).long())


        # pred_one = torch.gather(pred, dim=-1, index=(qshft_adjusted).long())  # [bs,l,1]

        target = datas[2]
        target = target[:, 1:, :]
        # pred_one = pred_one[:, :-1, :]
        target_1d = target.reshape(-1, 1)  # [batch_size * seq_len, 1]  [bs*199]
        mask = target_1d.ge(1)  # [batch_size * seq_len, 1]
        pred_1d = pred_one.reshape(-1, 1)  # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)  # [batch_size * seq_len - be masked, 1]
        filtered_target = torch.masked_select(target_1d, mask) - 1

        pred = self.sig(filtered_pred)


        # loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target.float())
        loss = self.crossEntropy(pred, filtered_target.float())

        return loss, pred, filtered_target.float()

        # batch = pred.squeeze(-1)
        # target = datas[2].squeeze(-1)
        # truth = target - 1
        # loss = 0.0
        # prediction = torch.tensor([], device=self.device)
        # ground_truth = torch.tensor([], device=self.device)
        # for student in range(batch.shape[0]):
        #     p = self.sig(batch[student][:-1])
        #     a = truth[student][1:].float()
        #
        #     '''看学生做的是哪一道题'''
        #     # delta = batch[student][:, :self.q] + batch[student][:, self.q:]  # [50,q]
        #     '''前49pred * 后49具体题目位置'''
        #     # temp = pred[student][:self.max_step - 1].mm(delta[1:].t())
        #     '''index: 0 ~ MAX_STEP-2'''
        #     # index = torch.tensor([[i for i in range(self.max_step - 1)]], dtype=torch.long, device=self.device)
        #     # '''提取[49, 49]对角线上的元素，因为这才是对应题目的预测值，其他位置都是无效信息'''
        #     # p = temp.gather(0, index)[0]
        #     # '''后49的1或0 ans'''
        #     # a = (((batch[student][:, 0:self.q] -
        #     #        batch[student][:, self.q:]).sum(1) + 1) //
        #     #      2)[1:]  # [49]
        #     #
        #
        #
        #     for i in range(len(p) - 1, -1, -1):
        #         if a[i] >= 0:
        #             p = p[:i + 1]
        #             a = a[:i + 1]
        #             found_valid = True
        #             break
        #
        #     loss += self.crossEntropy(p, a)
        #     prediction = torch.cat([prediction, p])
        #     ground_truth = torch.cat([ground_truth, a])
        #
        # return loss, prediction, ground_truth



def train_epoch(model, trainLoader, optimizer, loss_func, device):
    # global loss
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch = batch.to(device)
        # shape of a batch:[batch_size, max_step, 2 * ques]
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
    # performance(ground_truth, prediction)
    print('loss:', loss.item())
    return performance(ground_truth, prediction)
