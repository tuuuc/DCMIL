import pandas as pd
import argparse
import utils.c1_core_utils as c1
import utils.c2_core_utils as c2
import time
from sklearn.model_selection import train_test_split
import os

parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('--gpu', type=str, default='0', help='GPU')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--dataset', default='COAD', type=str)
parser.add_argument('--fold', default=0, type=int, help='cross validation')

####################### Curriculum 1 #########################
parser.add_argument('--training', type=int, default=1, help='Curriculum I training')
parser.add_argument('--encoding', type=int, default=1, help='Encoding instance')

parser.add_argument('--early_stopping', type=int, default=5)
parser.add_argument('--year', type=int, default=3)
parser.add_argument('--iter', type=int, default=1)

parser.add_argument('--total_epoch', type=int, default=250, help='The number of epoch')
parser.add_argument('--batch_size', type=int, default=12)

parser.add_argument('--pt_batch_size', type=int, default=128, help='The batch size of pretraining')
parser.add_argument('--pt_epoch', type=int, default=200, help='The epoch number of pretraining')
parser.add_argument('--pt_lr', default=1e-4, type=list, help='The learning rate of pretraining')
parser.add_argument('--warm_epoch', type=int, default=10)

parser.add_argument('--lr', default=1e-5, type=list, help='The learning rate')
parser.add_argument('--beta_r', default=1e-1, type=float, help='The weight coefficient of pairwise ranking loss')
parser.add_argument('--beta_omega', default=1e-5, type=float, help='The weight coefficient of structural loss')
parser.add_argument('--save_path', default='./result/c1/', type=str)
parser.add_argument('--inst_path', default='./data', type=str)

####################### Curriculum 2 #########################
parser.add_argument('--training_c2', type=int, default=1, help='Curriculum II training')
parser.add_argument('--nepochs_c2', type=int, default=1000, help='The maxium number of epochs to train')
parser.add_argument('--iters_c2', type=list, default=64)
parser.add_argument('--early_stopping_c2', type=int, default=50)
parser.add_argument('--k', type=int, default=30, help='top NB')
parser.add_argument('--dropout_c2', default=0.5, type=float)
parser.add_argument('--lr_c2', default=1e-5, type=float, help='The learning rate (default: Adam: 1e-3 | SGD: 1e-4~1e-5)')
parser.add_argument('--beta_adc', default=1e-1, type=float, help='The weight coefficient of absolute distance constraint')
parser.add_argument('--beta_tcl', default=1, type=float, help='The weight coefficient of triple-tier contrastive learning loss')
parser.add_argument('--beta_s', default=1e-4, type=float, help='The weight coefficient of sparseness loss')
parser.add_argument('--save_path_c2', default='./result/c2/', type=str)

args = parser.parse_args()

def main(args):
    if args.dataset in ['BLCA', 'COAD', 'HNSC', 'LIHC', 'OV', 'STAD']:
        args.year = 3
    elif args.dataset in ['BRCA', 'KIRC', 'LUAD', 'LUSC', 'THCA', 'UCEC']:
        args.year = 5
    for i in range(5):
        args.fold = i
        args.pt_epoch = 200 ## pretraining epoch
        print(f'dataset:{args.dataset}, fold:{args.fold}')
        data_path = f'./clinical/surv_{args.year}y/{args.dataset}_{args.year}y/data_{args.fold}.xlsx'
        train_df = pd.read_excel(data_path, sheet_name='train', names=['id','time','status','class'])
        train_pid, val_pid, _, _ = train_test_split(train_df,train_df['class'],test_size=0.2,random_state=23)
        test_df = pd.read_excel(data_path, sheet_name='test', names=['id','time','status','class'])
        
        ####################### Curriculum 1 #########################
        if args.training:
            c1.train(train_pid, val_pid, args=args)
            
        if args.encoding:
            c1.normal_embedding(args)
            if os.path.exists(os.path.join(args.save_path,args.dataset,f'fold_{args.fold}.xlsx')):
                print('finished!')
            else:
                c1.inst_encoding(train_df, test_df, args=args)

        ####################### Curriculum 2 #########################

        if args.training_c2:         

            data_path = args.save_path+'{}/fold_{}.xlsx'.format(args.dataset, args.fold)
            train_df = pd.read_excel(data_path, sheet_name='train')
            train_pid, val_pid = train_df[train_df[0].isin(train_pid['id'])], train_df[train_df[0].isin(val_pid['id'])]
            test_df = pd.read_excel(data_path, sheet_name='test')  
            print(f'dataset:{args.dataset}, fold:{args.fold}')
            c2.train(train_pid, val_pid, args)
            c2.evaluation(test_df, args)
            
    return

if __name__ == "__main__":
    print('beginning')
    start = time.time()
    main(args)
    end = time.time()
    print("finished!")
    print('Spending Time: %f seconds' % (end - start))
