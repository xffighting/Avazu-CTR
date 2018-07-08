import pandas as pd
import os
import time
import threadpool

# 多线程处理
def fea_countsum(fea): 
    
    day=['141021','141022','141023','141024','141025','141026','141027','141028','141029','141030']
    hour=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    dh=[]
    for d in day:
        for h in hour:
            dh.append(d+h)
            
    for i,c in enumerate(dh):
        
        path = "/media/leikun/programs/AI100/AI100-Final/Input/train/trainhour/count"+c
            
        fi =path+"/"+fea+".csv"
        if i==0:
            df = pd.read_csv(fi,usecols=[fea,'count'])
            df.rename(columns={'count':c}, inplace = True)
        else:
            df1 = pd.read_csv(fi,usecols=[fea,'count'])
            df1.rename(columns={'count':c}, inplace = True)
            
            df=pd.merge(df,df1,how='outer')
    
    df = df.fillna(0) # 把nan的值写为0，将各列设置为int
    for c in dh:
        df[c] = df[c].astype('int')
        
    path = "/media/leikun/programs/AI100/AI100-Final/Input/train/trainhour/summary"
    if not os.path.exists(path):
        os.makedirs(path)
        
    fo = path + "/" + fea + ".csv"
    df.to_csv(fo, mode='w',index=False)
    
    # df=df.set_index(fea) ；dc=df.to_dict()
    # 调用的时候，将特征值设置为行索引，那么某个特征值Vi，对应某个时间hi的出现次数为
    # count=dc[hi][vi]

    print(fea+" count summary processed")   

if __name__ == '__main__': 

    start0 = time.time()

    # 统计特征在每个时段出现的次数,除了device_id/device_ip,
    fea_count = ['C1', 'banner_pos',
            'site_id', 'site_domain', 'site_category',
            'app_id', 'app_domain', 'app_category',
            'device_model', 'device_type', 'device_conn_type',
            'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21','user_id','ad_id']

    pool = threadpool.ThreadPool(6) # 同时开线程数

    requests = threadpool.makeRequests(fea_countsum, fea_count) 

    [pool.putRequest(req) for req in requests]   

    pool.wait() # 等待所有线程结束 

    end0 = time.time()
    print("Total time use: %d sec." %(end0-start0))
