import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import torch
from torch import Tensor
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import random

def loss_r(class_prob, class_label, weights=None, delta=1e-3):
    loss = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')
    con_loss = nn.functional.relu(delta - (loss(class_prob[:,0,:], class_label) - loss(class_prob[:,1,:], class_label)))\
        + nn.functional.relu(delta - (loss(class_prob[:,1,:], class_label) - loss(class_prob[:,2,:], class_label)))
    return con_loss

def loss_fn(class_prob, class_label, weights=None):
    loss = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')
    return loss(class_prob, class_label)

def eval_ans(y_hat, true_label):
    return sum(y_hat == true_label)

def adjust_learning_rate(optimizer, lr):
    print('change learning rate')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class SPLLoss(nn.NLLLoss):
    def __init__(self, *args, **kwargs):
        super(SPLLoss, self).__init__(*args, **kwargs)
        self.if_perform = False
        self.threshold = 1
        self.growing_factor = 1.1
        self.v = []
        self.relu = nn.ReLU()
        self.delta = 1e-3

    def forward(self, input: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        scale = input.shape[1]
        if scale == 1:
            super_loss = loss_fn(input[:,0,:], target, weights)
            if self.if_perform:       
                v = self.spl_loss(super_loss)
                self.v += v.cpu().tolist()
                return (super_loss * v).mean()
            else:
                return super_loss.mean()

        else:
            loss_0, loss_1, loss_2 = loss_fn(input[:,0,:], target, weights),\
                                       loss_fn(input[:,1,:], target, weights),\
                                       loss_fn(input[:,2,:], target, weights)
            super_loss =  (loss_0 + loss_1 + loss_2) / 3

            con_loss = self.relu(self.delta - (loss_0 - loss_1)) + self.relu(self.delta - (loss_1 - loss_2))
    
            if self.if_perform:       
                v = self.spl_loss(super_loss)
                self.v += v.cpu().tolist()
                return (super_loss * v).mean(), (con_loss * v).mean()
            else:
                return super_loss.mean(), con_loss.mean()
        
    def increase_threshold(self):
        self.threshold *= self.growing_factor
        if len(self.v):
            if min(self.v): 
                print('stop pace learning')
                self.if_perform = False
            self.v = []
        
    def spl_loss(self, super_loss):
        v = super_loss < self.threshold
        return v.int()
    
def _neg_partial_log(prediction, T, E, device):
    current_batch_len = len(prediction)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.to(device)
    train_ystatus = torch.FloatTensor(E).to(device)

    theta = prediction.reshape(-1)
    exp_theta = torch.exp(theta)

    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    return loss_nn

def roc(data):

    _list = data.values.tolist()
    T = 3 * 12
    k = 0
    _copy = _list[:]
    for i in _copy:
        if i[0] <= T and i[1]==0:
            _list.pop(k) 
        else:
            k = k + 1
    _list = np.array(_list)
    y = (_list[:,0] < T).astype('int')
    status = _list[:,1]
    x = _list[:,2]

    fpr, tpr, thresholds = roc_curve(y, x)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def get_total_acc(y_patient):
    n = []
    for name in y_patient:
        n.append(np.mean(y_patient[name],axis=0) >= 0.5)
    return np.mean(n,axis=0)

def cox_log_rank(hazards, labels, survtime_all):
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    survtime_all = survtime_all
    idx = hazards_dichotomize == 0
    labels = labels
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)

def calculate_cindex(hazards, labels, survtime_all):
    labels = labels
    hazards = hazards.cpu().numpy().reshape(-1)
    label = []
    hazard = []
    surv_time = []
    for i in range(len(hazards)):
        if not np.isnan(hazards[i]):
            label.append(labels[i])
            hazard.append(hazards[i])
            surv_time.append(survtime_all[i])

    new_label = np.asarray(label)
    new_hazard = np.asarray(hazard)
    new_surv = np.asarray(surv_time)

    return (concordance_index(new_surv, -new_hazard, new_label))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


