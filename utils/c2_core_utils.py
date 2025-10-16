import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import random
import os
from utils.Early_Stopping import EarlyStopping
from utils.metrics import _neg_partial_log, cox_log_rank, calculate_cindex, roc
from models.model import SurvModel, TCLModel

####################### Curriculum 2 #########################

def train(train_df, val_df, args):
    device = torch.device(f'cuda:{args.gpu}')
    model = SurvModel(k = args.k, dropout = args.dropout_c2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_c2, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50, verbose=True)    

    model_save_path = f'{args.save_path_c2 + args.dataset}/models'
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(f'{args.save_path_c2 + args.dataset}/results', exist_ok=True)
    model_save_path += f'/c2_fold_{args.fold}.pth'

    early_stopping = EarlyStopping(model_path=model_save_path, patience=args.early_stopping_c2, verbose=True)
    
    train_list, val_list = train_df.values.tolist(), val_df.values.tolist()
    dir_pt = f'{args.save_path + args.dataset}/tumor_fold_{args.fold}/'
    dir_normal_pt = f'{args.save_path + args.dataset}/normal_fold_{args.fold}/'
    
    train_data = []
    for data in train_list:
        try:
            fea = torch.load(f'{dir_pt+data[0]}.pt', map_location=device)
        except:
            continue
        if fea.shape[0] < args.k: continue
        class_label = data[-1] if data[-2] == 2 else data[-2]
        bag_list = [fea, data[1], data[2], class_label]
        train_data.append(bag_list)

    val_data = []
    for data in val_list:
        try:
            fea = torch.load(f'{dir_pt+data[0]}.pt', map_location=device)
        except:
            continue
        if fea.shape[0] < args.k: continue
        bag_list = [fea, data[1], data[2]]
        val_data.append(bag_list)

    normal_data = []
    for i in os.listdir(dir_normal_pt):
        fea = torch.load(dir_normal_pt+i, map_location=device)
        if fea.shape[0] < 2: continue
        normal_data.append(fea)

    for epoch in range(args.nepochs_c2):
        train_epoch(epoch, args.iters_c2, model, optimizer, train_data, normal_data, args)
        valid_loss, val_ci, val_p, val_auc = prediction(model, val_data, args)
        scheduler.step()
    
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break  
            
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    valid_loss, val_ci, val_p, val_auc = prediction(model, val_data, args)

    return

def train_epoch(epoch, iters, model, optimizer, train_data, normal_data, args, measure=1, verbose=1):
    device = torch.device(f'cuda:{args.gpu}')
    tcl_model = TCLModel().to(device)
    model.train()
    tcl_model.train()
    # compiled_model = torch.compile(model)
    # compiled_tcl_model = torch.compile(tcl_model)

    survtime_iter, status_iter, survtime_all, status_all = [], [], [], []
    z, z_bar, z_normal, hazard = [], [], [], []

    iter = 0

    loss_nn_all = []
    random.shuffle(train_data)
    random.shuffle(normal_data)
    n_normal = len(normal_data)
    tbar = tqdm(train_data, desc='\r')

    for i_batch, data in enumerate(tbar):
        # loading data
        bag_tensor, time, status, h = data
        bag_tensor = bag_tensor.to(device)
        normal_tensor = normal_data[i_batch % n_normal].to(device)
        hazard.append(h)
        survtime_iter.append(time/30.0)
        status_iter.append(status)
        # ===================forward=====================
        y_pred, z_temp, z_bar_temp, z_normal_temp = model(bag_tensor, normal_tensor, training=True)

        z.append(z_temp)
        z_bar.append(z_bar_temp)
        z_normal.append(z_normal_temp)

        if iter == 0:
            y_pred_iter = y_pred
        else:
            y_pred_iter = torch.cat([y_pred_iter, y_pred])
        iter += 1

        if iter % iters == 0 or i_batch == len(train_data)-1:
            if np.max(status_iter) == 0:
                print("encounter no death in a batch, skip")
                continue

            optimizer.zero_grad()
            # =================== Cox loss =====================
            loss_cox = _neg_partial_log(y_pred_iter, np.asarray(survtime_iter), np.asarray(status_iter), device)
            
            # =================== TCL & ADC loss =====================
            hazard = torch.FloatTensor(hazard).to(device)
            z = torch.cat(z)
            z_bar = torch.cat(z_bar)
            z_low = z[hazard==0]
            z_high = z[hazard==1]
            z_bar = torch.cat([z_bar[hazard==0],z_bar[hazard==1]])
            z_normal = torch.cat(z_normal)
            
            loss_tcl, loss_adc = tcl_model(z_low=z_low, z_high=z_high, z_bar=z_bar, z_normal=z_normal, device=device)

            # =================== CSA loss =====================
            loss_s = None, None
            for param in model.self_attention.to_qkv.parameters():
                if loss_s is None:
                    loss_s = torch.abs(param).sum()
                else:
                    loss_s = loss_s + torch.abs(param).sum()
            
            # =================== total loss =====================   
            loss = loss_cox + args.beta_tcl * loss_tcl + args.beta_adc * loss_adc + args.beta_s * loss_s

            
    # ===================backward====================
            loss.backward()
            optimizer.step()

            y_pred_iter = None
            survtime_iter, status_iter = [], []
            loss_nn_all.append(loss.data.item())
            iter = 0
            z, z_bar, z_normal, hazard = [], [], [], []

        # ===================measure=====================
        if i_batch == 0:
            y_pred_all = y_pred[:,-1].detach().cpu()
        else:
            y_pred_all = torch.cat([y_pred_all, y_pred[:,-1].detach().cpu()])
        survtime_all.append(time)
        status_all.append(status)

    if measure:
        pvalue_pred = cox_log_rank(y_pred_all, np.asarray(status_all), np.asarray(survtime_all))
        c_index = calculate_cindex(y_pred_all, np.asarray(status_all), np.asarray(survtime_all))

        if verbose > 0:
            print("Epoch: {}, loss_nn: {}".format(epoch, np.mean(loss_nn_all)))
            print('[Training]\t loss (nn):{:.4f}'.format(np.mean(loss_nn_all)),
                  'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))

def prediction(model, queryloader, args, testing=False):
    device = torch.device(f'cuda:{args.gpu}')
    model.eval()

    status_all, survtime_all = [], []

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(queryloader):

            bag_tensor, time, status = sampled_batch

            bag_tensor = bag_tensor.to(device)
                
            survtime_all.append(time/30.0)
            status_all.append(status)

        # ===================forward=====================
            y_pred = model(bag_tensor)
            if i_batch == 0:
                y_pred_all = y_pred[:,-1]
            else:
                y_pred_all = torch.cat([y_pred_all, y_pred[:,-1]])

    survtime_all = np.asarray(survtime_all)
    status_all = np.asarray(status_all)

    loss = _neg_partial_log(y_pred_all, survtime_all, status_all, device)    

    pvalue_pred = cox_log_rank(y_pred_all.data, status_all, survtime_all)
    c_index = calculate_cindex(y_pred_all.data, status_all, survtime_all)

    csv_data = pd.DataFrame({'time':survtime_all,'event':status_all,'risk':np.squeeze(y_pred_all.cpu().numpy())})
    auc = roc(csv_data)

    if not testing:
        print('[val]\t loss (nn):{:.4f}'.format(loss.data.item()),
                      'c_index: {:.4f}, p-value: {:.3e}, auc: {:.3e}'.format(c_index, pvalue_pred, auc))
    else:
        print('[testing]\t loss (nn):{:.4f}'.format(loss.data.item()),
              'c_index: {:.4f}, p-value: {:.3e}, auc: {:.3e}'.format(c_index, pvalue_pred, auc))

        csv_data.to_csv(f'{args.save_path_c2}{args.dataset}/results/risk_{args.fold}.csv') 

    return loss.data.item(), c_index, pvalue_pred, auc

def evaluation(test_df, args):
    test_list = test_df.values.tolist()

    device = torch.device(f'cuda:{args.gpu}')

    model = SurvModel(k = args.k, dropout = args.dropout_c2).to(device)
    model_path = f'{args.save_path_c2 + args.dataset}/models/c2_fold_{args.fold}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    dir_pt = f'{args.save_path + args.dataset}/tumor_fold_{args.fold}/'

    test_data = []
    for data in test_list:
        try:
            fea = torch.load(f'{dir_pt+data[0]}.pt')
        except:
            continue
        if fea.shape[0] < args.k: continue
        bag_list = [fea, data[1], data[2]]
        test_data = test_data + [bag_list]
    
    _, c_index, p_value, auc = prediction(model, test_data, args, testing=True) 
    with open(f'{args.save_path_c2+args.dataset}/result_{args.fold}.txt','a')as fr:
        fr.write('fold:{}'.format(str(args.fold)) + '\n')  
    return 