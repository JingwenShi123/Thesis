
"""Implements supervised learning training procedures."""
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import time
from eval_scripts.performance import AUPRC, f1_score, accuracy, eval_affect
from eval_scripts.complexity import all_in_one_train, all_in_one_test
from eval_scripts.robustness import relative_robustness, effective_robustness, single_plot
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
softmax = nn.Softmax()
import sklearn.metrics
import numpy as np
import seaborn as sns


def ptsort(tu):
    return tu[0]


def AUPRC(pts):
    true_labels = [int(x[1]) for x in pts]
    predicted_probs = [x[0] for x in pts]
    return sklearn.metrics.average_precision_score(true_labels, predicted_probs)


def f1_score(truth, pred, average):
    return sklearn.metrics.f1_score(truth.cpu().numpy(), pred.cpu().numpy(), average=average)


def accuracy(truth, pred):
    return sklearn.metrics.accuracy_score(truth.cpu().numpy(), pred.cpu().numpy())


def eval_affect(truths, results, exclude_zero=True):
    if type(results) is np.ndarray:
        test_preds = results
        test_truth = truths
    else:
        test_preds = results.cpu().numpy()
        test_truth = truths.cpu().numpy()

    non_zeros = np.array([i for i, e in enumerate(
        test_truth) if e != 0 or (not exclude_zero)])

    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)



    # 计算 F1 分数
    f1 = sklearn.metrics.f1_score(binary_truth, binary_preds, average='binary')
    # 计算准确率
    accuracy = sklearn.metrics.accuracy_score(binary_truth, binary_preds)

    return f1, accuracy
    # return sklearn.metrics.accuracy_score(binary_truth, binary_preds)



class MMDL(nn.Module):
    """Implements MMDL classifier."""

    def __init__(self, encoders, fusion, head, has_padding=False):
        """Instantiate MMDL Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
        """
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs):
        """Apply MMDL to Layer Input.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i](
                    [inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))
        self.reps = outs
        if self.has_padding:

            if isinstance(outs[0], torch.Tensor):
                out = self.fuse(outs)
            else:
                out = self.fuse([i[0] for i in outs])
        else:
            out = self.fuse(outs)
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            return self.head([out, inputs[1][0]])
        return self.head(out)


def deal_with_objective(objective, pred, truth, args):
    """Alter inputs depending on objective function, to deal with different objective arguments."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if type(objective) == nn.CrossEntropyLoss:
        if len(truth.size()) == len(pred.size()):
            truth1 = truth.squeeze(len(pred.size()) - 1)
        else:
            truth1 = truth
        return objective(pred, truth1.long().to(device))
    elif type(objective) == nn.MSELoss or type(objective) == nn.modules.loss.BCEWithLogitsLoss or type(
            objective) == nn.L1Loss:
        return objective(pred, truth.float().to(device))
    else:
        return objective(pred, truth, args)


def train(
        encoders, fusion, head, train_dataloader, valid_dataloader, total_epochs, additional_optimizing_modules=[], is_packed=False,
        early_stop=False, optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None, input_to_float=True, clip_val=8,
        track_complexity=True):
    """
    Handle running a simple supervised training loop.

    :param encoders: list of modules, unimodal encoders for each input modality in the order of the modality input data.
    :param fusion: fusion module, takes in outputs of encoders in a list and outputs fused representation
    :param head: classification or prediction head, takes in output of fusion module and outputs the classification or prediction results that will be sent to the objective function for loss calculation
    :param total_epochs: maximum number of epochs to train
    :param additional_optimizing_modules: list of modules, include all modules that you want to be optimized by the optimizer other than those in encoders, fusion, head (for example, decoders in MVAE)
    :param is_packed: whether the input modalities are packed in one list or not (default is False, which means we expect input of [tensor(20xmodal1_size),(20xmodal2_size),(20xlabel_size)] for batch size 20 and 2 input modalities)
    :param early_stop: whether to stop early if valid performance does not improve over 7 epochs
    :param task: type of task, currently support "classification","regression","multilabel"
    :param optimtype: type of optimizer to use
    :param lr: learning rate
    :param weight_decay: weight decay of optimizer
    :param objective: objective function, which is either one of CrossEntropyLoss, MSELoss or BCEWithLogitsLoss or a custom objective function that takes in three arguments: prediction, ground truth, and an argument dictionary.
    :param auprc: whether to compute auprc score or not
    :param save: the name of the saved file for the model with current best validation performance
    :param validtime: whether to show valid time in seconds or not
    :param objective_args_dict: the argument dictionary to be passed into objective function. If not None, at every batch the dict's "reps", "fused", "inputs", "training" fields will be updated to the batch's encoder outputs, fusion module output, input tensors, and boolean of whether this is training or validation, respectively.
    :param input_to_float: whether to convert input to float type or not
    :param clip_val: grad clipping limit
    :param track_complexity: whether to track training complexity or not
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MMDL(encoders, fusion, head, has_padding=is_packed).to(device)
    train_losses = []
    val_losses = []
    def _trainprocess():
        additional_params = []
        for m in additional_optimizing_modules:
            additional_params.extend(
                [p for p in m.parameters() if p.requires_grad])
        op = optimtype([p for p in model.parameters() if p.requires_grad] +
                       additional_params, lr=lr, weight_decay=weight_decay)
        bestvalloss = 10000
        patience = 0

        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp

        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                if is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        model.train()
                        out = model([[_processinput(i).to(device)
                                      for i in j[0]], j[1]])

                else:
                    model.train()
                    out = model([_processinput(i).to(device)
                                 for i in j[:-1]])

                if not (objective_args_dict is None):
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model
                loss = deal_with_objective(
                    objective, out, j[-1], objective_args_dict)

                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()
            trainloss = totalloss /totals
            train_losses.append(trainloss.item())
            print("Epoch  " +str(epoch ) +" train loss:  " +str(trainloss))
            validstarttime = time.time()
            if validtime:
                print("train total:  " +str(totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                true = []
                pts = []
                for j in valid_dataloader:
                    if is_packed:
                        out = model([[_processinput(i).to(device)
                                      for i in j[0]], j[1]])
                    else:
                        out = model([_processinput(i).to(device)
                                     for i in j[:-1]])

                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False
                    loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict)
                    totalloss += loss *len(j[-1])

                    true.append(j[-1])
                    if auprc:
                        # pdb.set_trace()
                        sm = softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss /totals
            val_losses.append(valloss.item())
            print("Epoch  " +str(epoch ) +" valid loss:  " +str(valloss.item()))
            if valloss < bestvalloss:
                patience = 0
                bestvalloss = valloss
                print("Saving Best")
                torch.save(model, save)
            else:
                patience += 1
            if early_stop and patience > 7:
                break
            if auprc:
                print("AUPRC:  " +str(AUPRC(pts)))
            validendtime = time.time()
            if validtime:
                print("valid time:   " +str(validendtime -validstarttime))
                print("Valid total:  " +str(totals))
    if track_complexity:
        all_in_one_train(_trainprocess, [model ] +additional_optimizing_modules)
    else:
        _trainprocess()
    return train_losses, val_losses

def single_test(
        model, test_dataloader, is_packed=False,
        criterion=nn.CrossEntropyLoss(), auprc=False, input_to_float=True):
    """Run single test for model.

    Args:
        model (nn.Module): Model to test
        test_dataloader (torch.utils.data.Dataloader): Test dataloader
        is_packed (bool, optional): Whether the input data is packed or not. Defaults to False.
        criterion (_type_, optional): Loss function. Defaults to nn.CrossEntropyLoss().
        task (str, optional): Task to evaluate. Choose between "classification", "multiclass", "regression", "posneg-classification". Defaults to "classification".
        auprc (bool, optional): Whether to get AUPRC scores or not. Defaults to False.
        input_to_float (bool, optional): Whether to convert inputs to float before processing. Defaults to True.
    """

    def split2(data):
        result = np.where(data < 0, 0, 1)
        return result

    def split7(data):
        # 对数据进行条件操作
        result = np.where((data >= -3) & (data < -2), 1,
                          np.where((data >= -2) & (data < -1), 2,
                                   np.where((data >= -1) & (data < 0), 3,
                                            np.where((data > 0) & (data <= 1), 4,
                                                     np.where((data > 1) & (data <= 2), 5,
                                                              np.where((data > 2) & (data <= 3), 6, data))))))
        return result
    def _processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp
    with torch.no_grad():
        totalloss = 0.0
        pred = []
        true = []
        pts = []
        all_oute = []
        for j in test_dataloader:
            model.eval()
            if is_packed:
                out = model([[_processinput(i).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                              for i in j[0]], j[1]])
            else:
                out = model([_processinput(i).float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                             for i in j[:-1]])

            if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss or type(criterion) == torch.nn.MSELoss:
                loss = criterion(out, j[-1].float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

            elif type(criterion) == nn.CrossEntropyLoss:
                if len(j[-1].size()) == len(out.size()):
                    truth1 = j[-1].squeeze(len(out.size() ) -1)
                else:
                    truth1 = j[-1]
                loss = criterion(out, truth1.long().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            else:
                loss = criterion(out, j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            totalloss += loss *len(j[-1])
            prede = []
            oute = out.cpu().numpy().tolist()  # 预测的结果
            all_oute.extend(oute)
            for i in oute:  # 转换成1，-1，0
                if i[0] > 0:
                    prede.append(1)
                elif i[0] < 0:
                    prede.append(-1)
                else:
                    prede.append(0)

            #prede = prede[0]
            pred.append(torch.LongTensor(prede))

            true.append(j[-1])
            if auprc:
                # pdb.set_trace()
                sm = softmax(out)
                pts += [(sm[i][1].item(), j[-1][i].item())
                        for i in range(j[-1].size(0))]
        if oute:
            pred_reg = torch.Tensor(all_oute)  # 回归预测值
        if pred:
            pred = torch.cat(pred, 0)  # 分类预测结果

        prede = torch.tensor(prede)
        prede = prede.view(-1, 1)

        true = torch.cat(true, 0)
        totals = true.shape[0]
        testloss = totalloss /totals
        if auprc:
            print("AUPRC:  " +str(AUPRC(pts)))

        trueposneg = true  # 真实值

        Acc7 = accuracy_score(split7(trueposneg).astype(int), split7(pred_reg).astype(int))
        corr, _ = pearsonr(trueposneg.squeeze(), pred_reg.squeeze())
        mae = np.array(torch.sum(torch.abs(trueposneg - pred_reg)) / len(pred_reg))
        f1s, accs = eval_affect(trueposneg, prede)
        f12, acc2 = eval_affect(trueposneg, prede, exclude_zero=False)
        print('------------------------------------------------------')
        print("MSE: " + str(testloss.item()))
        print("Acc7: " + str(Acc7))
        print("Corr: " + str(corr))
        print("MAE: " + str(mae))
        print("F1 " + str(f1s) + ',' + str(f12))
        print("Acc2:  " +str(accs) + ',' + str(acc2))

        # 示例数据
        true_labels = split7(trueposneg).astype(int)  # 真实标签
        predicted_labels = split7(pred_reg).astype(int)  # 预测标签

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix,cmap="YlGnBu_r",fmt="d",annot=True)
        plt.title('Confusion Matrix - 7')
        plt.show()

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(split2(trueposneg).astype(int), split2(prede).astype(int))
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix,cmap="YlGnBu_r",fmt="d",annot=True)

        plt.title('Confusion Matrix - 7')
        plt.show()



def test(
        model, test_dataloaders_all, dataset='default', method_name='My method', is_packed=False, criterion=nn.CrossEntropyLoss(), auprc=False, input_to_float=True, no_robust=False):
    """
    Handle getting test results for a simple supervised training loop.

    :param model: saved checkpoint filename from train
    :param test_dataloaders_all: test data
    :param dataset: the name of dataset, need to be set for testing effective robustness
    :param criterion: only needed for regression, put MSELoss there
    """
    if no_robust:
        def _testprocess():
            single_test(model, test_dataloaders_all, is_packed,
                        criterion, auprc, input_to_float)
        all_in_one_test(_testprocess, [model])
        return

    def _testprocess():
        single_test(model, test_dataloaders_all[list(test_dataloaders_all.keys())[
            0]][0], is_packed, criterion, auprc, input_to_float)
    all_in_one_test(_testprocess, [model])
    for noisy_modality, test_dataloaders in test_dataloaders_all.items():
        print("Testing on noisy data ({})...".format(noisy_modality))
        robustness_curve = dict()
        for test_dataloader in tqdm(test_dataloaders):
            single_test_result = single_test(
                model, test_dataloader, is_packed, criterion, auprc, input_to_float)
            for k, v in single_test_result.items():
                curve = robustness_curve.get(k, [])
                curve.append(v)
                robustness_curve[k] = curve
        for measure, robustness_result in robustness_curve.items():
            robustness_key = '{} {}'.format(dataset, noisy_modality)
            print("relative robustness ({}, {}): {}".format(noisy_modality, measure, str(
                relative_robustness(robustness_result, robustness_key))))
            if len(robustness_curve) != 1:
                robustness_key = '{} {}'.format(robustness_key, measure)
            print("effective robustness ({}, {}): {}".format(noisy_modality, measure, str(
                effective_robustness(robustness_result, robustness_key))))
            fig_name = '{}-{}-{}-{}'.format(method_name,
                                            robustness_key, noisy_modality, measure)
            single_plot(robustness_result, robustness_key, xlabel='Noise level',
                        ylabel=measure, fig_name=fig_name, method=method_name)
            print("Plot saved as  " +fig_name)