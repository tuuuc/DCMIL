import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
 
 
datasets = ['KIRC','LUAD','LUSC','STAD','THCA','HNSC','LIHC','BRCA','BLCA','OV','UCEC','COAD']
#### 按年分类
years = [3,5,8,10]
name = ['id','time','status','class']
for dataset in datasets[:-2]:
    dirpath = './clinical/'
    data = pd.read_csv(dirpath+f'{dataset}_info.csv',usecols=name[:-1]).values
    for year in years:
        surv  = []
        for i,temp in enumerate(data):
            if temp[2]==-1:
                surv.append(list(data[i])+[2])
                continue
            if temp[2]==1 and int(temp[1]) < year*365:
                surv.append(list(data[i])+[1])
            elif int(temp[1]) >= year*365:
                surv.append(list(data[i])+[0])
            else:
                surv.append(list(data[i])+[2])
        df = pd.DataFrame(columns=name,data=surv)
        save_path = dirpath + f'surv_{year}y/{dataset}_{year}y'
        if not os.path.exists(save_path): os.makedirs(save_path)
        df.to_csv(save_path + '.csv',index=None)
 
        x = df.values
        y = df['status'].values
 
        kf = StratifiedKFold(n_splits=5,random_state=23,shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(x, y)):
            res_train = pd.DataFrame(data=x[train_index])
            res_test = pd.DataFrame(data=x[test_index])
            writer = pd.ExcelWriter(save_path+f'/data_{i}.xlsx')
            eval('res_train').to_excel(excel_writer=writer, sheet_name='train', index=False)
            eval('res_test').to_excel(excel_writer=writer, sheet_name='test', index=False)
            writer.save()
            writer.close()
