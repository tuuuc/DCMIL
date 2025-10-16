import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import os
from tqdm import tqdm

import time
from utils.utils import *
from utils.metrics import *
from utils.Early_Stopping import EarlyStopping
from models.model import VitBranch, get_parameter_number
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

####################### Curriculum 1 #########################
    
def train(train_df, val_df, args):
    device = torch.device(f'cuda:{args.gpu}')
    torch.backends.cudnn.benchmark = True

    model_save_path = f'{args.save_path + args.dataset}/models'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model_save_path = model_save_path + '/c1_fold_{}.pth'.format(args.fold)
    spath = model_save_path.replace('.pth', f'_model.pth')
    spath_pre = spath.replace('.pth', f'_pre.pth')

    setup_seed(23)
        
    train_list, val_list = train_df.values.tolist(), val_df.values.tolist()
    tile_map = get_tile_map(args)

    dic = {}
    for id, tile_list in tile_map.items():
        if id[:12] not in dic.keys(): dic[id[:12]] = []
        dic[id[:12]].append([id, tile_list])

    train_inst, weights = build_inst_multi(dic, train_list, args)
    weights = weights.to(device)

    val_inst, _ = build_inst_multi(dic, val_list, args)

    train_inst = CustomImageDataset(train_inst)
    val_inst = CustomImageDataset(val_inst)

    train_inst = DataLoader(train_inst, batch_size=args.pt_batch_size,\
                                   shuffle=True, num_workers=args.workers,pin_memory=True)
    val_inst = DataLoader(val_inst, batch_size=args.pt_batch_size,\
                                shuffle=False, num_workers=args.workers,pin_memory=True)

    nepoch = args.total_epoch

    model = VitBranch().to(device)
    get_parameter_number(model)    
    scaler = GradScaler()
    criterion = SPLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.pt_lr, weight_decay = 1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)    
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.pt_lr/10,\
                                                  max_lr=args.pt_lr, step_size_up=30,\
                                                    step_size_down=20, cycle_momentum=False)
    # writer = SummaryWriter('./plot/tf_log')
    
    if os.path.exists(spath) and args.pt_epoch == 0:
        print('loading existing model')
        model.load_state_dict(torch.load(spath, map_location=device))
    elif os.path.exists(spath_pre): 
        print('loading existing pretraining model')
        model.load_state_dict(torch.load(spath_pre, map_location=device))
        
    early_stopping = EarlyStopping(model_path=spath, patience=args.early_stopping, verbose=True)
    
    for epoch in range(nepoch):
        print("\n Epoch: {}".format(epoch))

        if epoch == args.pt_epoch: 
            optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = 1e-5)
            train_inst.batch_sampler.batch_size = args.batch_size
            val_inst.batch_sampler.batch_size = args.batch_size
        if epoch == args.warm_epoch:
            criterion.if_perform = True
            criterion.threshold = train_loss * 0.8
            print('begin pace learning')
        elif criterion.if_perform and epoch > args.warm_epoch:
            criterion.increase_threshold()
            
        train_loss, train_acc = train_epoch(epoch, model, scaler, criterion, optimizer, train_inst, args, weights)

        if epoch < args.pt_epoch:
            scheduler.step()        
            if train_acc[0] > 0.7 or (epoch+1) == args.pt_epoch:
                if epoch == 0:
                    print('pretraining finished!')
                    return
                for s in range(2):
                    model.modules[s+1].vit.embeddings.patch_embeddings.load_state_dict(model.modules[0].vit.embeddings.patch_embeddings.state_dict())
                    _len = len(model.modules[0].vit.encoder.layer)
                    model.modules[s+1].vit.encoder.layer[:_len].load_state_dict(model.modules[0].vit.encoder.layer.state_dict())
                torch.save(model.state_dict(), spath_pre)
                print('save pretraining model')
                args.pt_epoch = epoch + 1
            continue

        valid_loss, val_acc = prediction(model, val_inst, args)
        
        # writer.add_scalar(f'result/Loss_train', train_loss, epoch)
        # writer.add_scalar(f'result/Loss_test', valid_loss, epoch)        
        # for j in range(3):
        #     writer.add_scalar(f'result/Accuracy_train{j}', train_acc[j], epoch)
        #     writer.add_scalar(f'result/Accuracy_test{j}', val_acc[j], epoch)    

        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
    torch.save(model.state_dict(), spath)
    return

def train_epoch(epoch, model, scaler, criterion, optimizer, train_inst, args, weights=None, measure=1):
    device = torch.device(f'cuda:{args.gpu}')
    
    model.train()
    # compiled_model = torch.compile(model)

    y_label = []
    y_patient = {}
    iter = 0
    loss_all = []
    
    begin = time.time()

    pt_epoch = True if epoch < args.pt_epoch else False
    
    for i, data in enumerate(tqdm(train_inst)):
        x1, x2, x3, label, names = data
        label = label.squeeze(1)
        x = [x1.to(device), x2.to(device), x3.to(device)]

        # ===================forward=====================
        with autocast():
            y_prob, y_hat = model(x, label, pt_epoch, device=device) # (batch,scale,2), (batch,scale)

        for b in range(len(names)):
            name = names[b]
            if name not in y_patient.keys():
                y_patient[name] = []
            y_patient[name].append(y_hat[b] == label[b].item())
            
        y_label = np.concatenate((y_label, label.numpy()), axis=0) if i else label.numpy()
        y_hat_all = np.concatenate((y_hat_all, y_hat), axis=0) if i else y_hat
        y_iter = torch.cat((y_iter, label)) if iter else label
        y_prob_iter = torch.cat((y_prob_iter, y_prob)) if iter else y_prob

        iter += 1

        if iter % args.iter == 0:
            optimizer.zero_grad()
            with autocast():
                if epoch < args.pt_epoch:
                    loss_cls = criterion(y_prob_iter, y_iter.to(device), weights)
                    loss = loss_cls
                else:
                    loss_cls, loss_r = criterion(y_prob_iter, y_iter.to(device), weights)

                    loss_omega = 0
                    for s in range(2):
                        param_em = zip(model.modules[s].vit.embeddings.patch_embeddings.parameters(), 
                                    model.modules[s+1].vit.embeddings.patch_embeddings.parameters())
                        param_transformer_1 = model.modules[s].vit.encoder.layer
                        param_transformer_2 = model.modules[s+1].vit.encoder.layer
                        _len = len(param_transformer_1)
                        param_tr = zip(param_transformer_1.parameters(),param_transformer_2[:_len].parameters())
                        for (p1,p2) in param_em:
                            loss_omega += torch.norm(p1-p2)
                        for (p1,p2) in param_tr:
                            loss_omega += torch.norm(p1-p2)

                    loss = loss_cls + args.beta_r * loss_r + args.beta_omega * loss_omega
        # ===================backward====================
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() 

            loss_all.append([loss.data.item(), loss_cls.data.item()])
            y_prob_iter = None
            y_iter = []
            iter = 0
            
            # torch.cuda.empty_cache()
            # gc.collect()

    end = time.time()
    print('time: {}s'.format(end-begin))

    if measure:
        num, scale = y_hat_all.shape
        y_label = np.asarray(y_label)
        pace_num = sum(criterion.v) if len(criterion.v) else num
        loss_all = np.mean(loss_all,0) * num / pace_num if pace_num else [0,0]
        train_acc = [eval_ans(y_hat_all[:,i], y_label) / num for i in range(scale)]
        total_train_acc = get_total_acc(y_patient)
        print(['self_pace number: {}, training_loss: {:.4f}, class_loss: {:.4f}'.format(pace_num, loss_all[0], loss_all[1])])
        print(['acc: {}'.format(train_acc), 'total_acc: {}'.format(total_train_acc)])
        return loss_all[1], total_train_acc
    return

@torch.no_grad()
def prediction(model, queryloader, args, verbose=1):
    device = torch.device(f'cuda:{args.gpu}')
    model.eval()
    y_patient = {}

    for i, data in enumerate(queryloader):
        x1, x2, x3, label, names = data
        label = label.squeeze(1).to(device)
        x = [x1.to(device), x2.to(device), x3.to(device)]

        # ===================forward=====================
        with autocast():
            y_prob, y_hat = model(x, y=None, device=device)

        for b in range(len(names)):
            name = names[b]
            if name not in y_patient.keys():
                y_patient[name] = []
            y_patient[name].append(y_hat[b] == label[b].item())

        y_hat_all = np.concatenate((y_hat_all, y_hat), axis=0) if i else y_hat
        y_label = torch.cat((y_label, label)) if i else label
        y_prob_all = torch.cat((y_prob_all, y_prob)) if i else y_prob
        
    with autocast():
        loss_cls = (loss_fn(y_prob_all[:,0,:], y_label)+loss_fn(y_prob_all[:,1,:], y_label)+loss_fn(y_prob_all[:,2,:], y_label)).mean().data.item()
        loss_r = loss_r(y_prob_all, y_label).mean().data.item()

    num = len(y_label)
    y_label = y_label.cpu().numpy()
    
    acc = [eval_ans(y_hat_all[:,i], y_label) / num for i in range(3)]
    total_acc = get_total_acc(y_patient)

    if verbose:
        print(['loss_cls: {:.4f}'.format(loss_cls), 'loss_r: {:.4f}'.format(loss_r)])
        print(['acc: {}'.format(acc), 'total_acc: {}'.format(total_acc)])
    
    return loss_cls, total_acc

def inst_encoding(train_df, test_df, args):
    setup_seed(23)

    print('loading model')
    device = torch.device(f'cuda:{args.gpu}')
    model_save_path = args.save_path+'{}/models/c1_fold_{}.pth'.format(args.dataset,args.fold)
    tile_map = get_tile_map(args)
    model = VitBranch().to(device)
    model.load_state_dict(torch.load(model_save_path.replace('.pth', '_model.pth'),map_location=device))

    save_path =  args.save_path+'{}/tumor_fold_{}/'.format(args.dataset,args.fold)
    if not os.path.exists(save_path):
        os.makedirs(save_path)   

    train_list, test_list = train_df.values.tolist(), test_df.values.tolist()

    dic = {}
    for id, tile_list in tile_map.items():
        if id[:12] not in dic.keys(): dic[id[:12]] = []
        dic[id[:12]].append([id, tile_list])

    train_res = []
    for data in train_list:
        name = data[0]
        if name not in dic.keys(): continue
        save_path_pt = save_path + f'{name}.pt'

        inst_list = build_inst_multi(dic, [data], args, prediction=True)
        if not len(inst_list): continue
        train_res.append(data+encoder_pt(model, inst_list, save_path_pt, args))
    
    test_res = []
    for data in test_list:
        name = data[0]
        if name not in dic.keys(): continue
        save_path_pt = save_path + f'{name}.pt'
        
        inst_list = build_inst_multi(dic, [data], args, prediction=True)
        if not len(inst_list): continue
        test_res.append(data+encoder_pt(model, inst_list, save_path_pt, args))

    train_res = pd.DataFrame(data=train_res)
    test_res = pd.DataFrame(data=test_res)

    res_path = args.save_path+'{}/'.format(args.dataset)
    with pd.ExcelWriter(res_path+'fold_{}.xlsx'.format(args.fold)) as writer:
        train_res.to_excel(writer, sheet_name='train', index=False)
        test_res.to_excel(writer, sheet_name='test', index=False)

    return

@torch.no_grad()
def encoder_pt(model, query, save_path, args):
    batch_size = 32
    device = torch.device(f'cuda:{args.gpu}')
    model.eval()

    class_label = query[0][-2]   

    query = CustomImageDataset(query)
    queryloader = DataLoader(query, batch_size=batch_size,\
                                shuffle=False, num_workers=args.workers,pin_memory=True)
    
    features, outs = [], 0
    
    for i, data in enumerate(queryloader):
        x1, x2, x3, _, _ = data
        x = [x1.to(device), x2.to(device), x3.to(device)]

        fea, pro, out = model.extract_fea(x, device=device) 

        features.append(fea)
        outs += (out.sum(1)>=2).sum()

    if class_label == 2:
        class_label = 1 if (outs / len(query)) >= 0.5 else 0

    features = torch.cat(features, 0)
    torch.save(features, save_path)
    return [class_label]

@torch.no_grad()
def normal_embedding(args):
    setup_seed(23)
    batch_size = 32
    device = torch.device(f'cuda:{args.gpu}')
    model_path = args.save_path+'{}/models/c1_fold_{}.pth'.format(args.dataset,args.fold)
    print('loading model')
    model = VitBranch().to(device)
    model.load_state_dict(torch.load(model_path.replace('.pth', '_model.pth'),map_location=device))
    model.eval()
    
    tile_dir_5x = f'{args.inst_path}/{args.dataset}_ms/normal/5x'
    tile_dir_10x = f'{args.inst_path}/{args.dataset}_ms/normal/10x'
    tile_dir_20x = f'{args.inst_path}/{args.dataset}_ms/normal/20x'
    save_path =  args.save_path+'{}/normal_fold_{}/'.format(args.dataset,args.fold)
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    
    tile_dir = os.listdir(f'{args.inst_path}/{args.dataset}_ms/normal/location')
    for i in tile_dir: # make tile ID lists for all slides
        i = i.split('.')[0]
        save_path_pt = save_path + f'{i}.pt'
        if os.path.exists(save_path_pt): continue

        tiles = os.listdir(f'{tile_dir_5x}/{i}')
        tiles.sort(key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        inst_list = [[f'{tile_dir_5x}/{i}/{tile_i}', f'{tile_dir_10x}/{i}/{tile_i}', f'{tile_dir_20x}/{i}/{tile_i}', 0, 0] for tile_i in tiles]

        query = CustomImageDataset(inst_list)
        queryloader = DataLoader(query, batch_size=batch_size,\
                                    shuffle=False, num_workers=args.workers,pin_memory=True)
        
        features = []
        
        for i, data in enumerate(queryloader):
            x1, x2, x3, _, _ = data
            x = [x1.to(device), x2.to(device), x3.to(device)]

            fea, _, _ = model.extract_fea(x, device=device) 
            features.append(fea)  
        features = torch.cat(features, 0)
        torch.save(features, save_path_pt)        
    return 
    