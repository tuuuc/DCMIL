import pandas as pd
import os
 
datasets = ['KIRC','LUAD','LUSC','STAD','THCA','HNSC','LIHC','BRCA','BLCA','OV','UCEC','COAD']
for k in range(len(datasets)):
    dirpath = './clinical/'
    frame = pd.read_csv(dirpath+'clinical.tsv', sep='\t')
    
    #### 删除相同的行
    data = frame.drop_duplicates(subset=['case_submitter_id'], keep='first', inplace=False)
    data.reset_index(drop=True, inplace=True)
     
    #### 筛选临床信息
    columns_dict = {"case_submitter_id":"id",
                    "age_at_index":"age",
                    "gender":"gender",
                    "race":"race",
                    "vital_status":"vital_status",
                    "days_to_death":"days_to_death",
                    "days_to_last_follow_up":"days_to_last_follow_up",
                    "ajcc_pathologic_stage":"tumor_stage",
                    'ajcc_pathologic_t':'tumor_T'
    }
     
    #### 列名重命名
    df = pd.DataFrame(data, columns=columns_dict.keys())
    df.rename(columns=columns_dict, inplace=True) 
     
    #### 遍历所有WSI文件
    def img_path(dirname):
        filter1 = [".mat"]
        img_path = []
        for maindir, subdir, file_name_list in os.walk(dirname):
            for filename in file_name_list:
                ext = os.path.splitext(filename)[1]
                if ext in filter1 and int(filename[13:15])<10:
                    img_path.append(filename[0:12])
        return img_path
    path=f'./data/{datasets[k]}_ms/tumor/location/'
    finished=img_path(path)
    print(len(finished))
     
    #### 提取生存信息
    status, time = [], []
    for ind,i in enumerate(df['vital_status']):
        if i == 'Alive' and df['days_to_last_follow_up'][ind]!="'--":
            status.append(0)
            time.append(df['days_to_last_follow_up'][ind])
        elif i == 'Dead' and df['days_to_death'][ind]!="'--":
            status.append(1)
            time.append(df['days_to_death'][ind])
        else:
            status.append(-1)
            time.append(df['days_to_last_follow_up'][ind])        
    df['time'] = time
    df['status'] = status
 
    #### 保留有.svs数据的临床数据
    inds = []
    for ind,i in enumerate(df['id']):
        if i not in finished or int(df['status'][ind]) == -1 or int(df['time'][ind]) <= 30:
            inds.append(ind)
    df.drop(df.index[inds],inplace=True)
 
    #### 将重命名之后的数据写入到文件
    filepath = dirpath+f'{datasets[k]}_info.csv'
    df_columns = pd.DataFrame([list(df.columns)])
    df_columns.to_csv(filepath, mode='w', header=False, index=0) 
    df.to_csv(filepath, mode='a', header=False, index=0)
