import random
import pandas as pd
import os
import time
import threadpool
import hashlib

def hashstr(str, nr_bins=1e6):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1

# 训练集
day=['141021','141022','141023','141024','141025','141026','141027','141028','141029','141030']
hour=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']

# 多线程处理
def fea_count(dh): 

    print("train"+dh+".csv processing ....")   

    fi = "/media/leikun/programs/AI100/AI100-Final/Input/train/trainhour/pn/train"+dh+".csv"
    df = pd.read_csv(fi)

    path = "/media/leikun/programs/AI100/AI100-Final/Input/train/trainhour/count"+dh
    if not os.path.exists(path):
        os.makedirs(path)

    df['user_id1'] = df['device_ip'].astype(str)+'+'+df['device_model'].astype(str)
    df['user_id'] = df['user_id1'].apply(lambda x : hashstr(x))
    df['user_id'] = df['user_id'].astype('int')

    df['ad_id1'] = df['C14'].astype(str)+'+'+df['C17'].astype(str)
    df['ad_id'] = df['ad_id1'].apply(lambda x : hashstr(x))
    df['ad_id'] = df['ad_id'].astype('int')

    df.drop('user_id1', axis=1, inplace=True)
    df.drop('ad_id1', axis=1, inplace=True)

    # 统计特征在每个时段出现的次数,除了device_id/device_ip,
    fea_count = ['C1', 'banner_pos',
          'site_id', 'site_domain', 'site_category',
          'app_id', 'app_domain', 'app_category',
           'device_model', 'device_type', 'device_conn_type',
          'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21','user_id','ad_id']


    dfo = pd.crosstab(df['user_id'],df['ad_id'])
    fo = path + "/" + "user_id2ad_id" + ".csv"
    dfo.to_csv(fo, mode='w',index=True)

    # for c in fea_count:

    #     count1 = pd.crosstab(df['hour'],df[c])
    #     a1=list(count1.columns)
    #     a2=list(count1.values[0])
    #     a={c : a1,
    #     "count" : a2}
    #     data1=pd.DataFrame(a)

    #     count2 = df['click'].groupby([df['click'], df[c]]).count()
    #     b=count2.xs(0, level=0)
    #     a1=list(b.index)
    #     a2=list(b)
    #     a={c : a1,
    #     "click0" : a2}
    #     data2=pd.DataFrame(a)

    #     b=count2.xs(1, level=0)
    #     a1=list(b.index)
    #     a2=list(b)
    #     a={c : a1,
    #     "click1" : a2}
    #     data3=pd.DataFrame(a)

    #     dfo=pd.merge(data1,data2,how='left')
    #     dfo=pd.merge(dfo,data3,how='left')

    #     dfo=dfo[[c,'count','click0','click1']]
    #     dfo = dfo.fillna(0)
    #     dfo['click0'] = dfo['click0'].astype('int')
    #     dfo['click1'] = dfo['click1'].astype('int')

    #     fo = path + "/" + c + ".csv"

    #     dfo.to_csv(fo, mode='w',index=False)

    print("train"+dh+".csv processed")        

if __name__ == '__main__': 

    start0 = time.time()

    param_list=[]
    for d in day:
        for h in hour:
            param_list.append(d+h)

    pool = threadpool.ThreadPool(6) # 同时开线程数

    requests = threadpool.makeRequests(fea_count, param_list) 

    [pool.putRequest(req) for req in requests]   

    pool.wait() # 等待所有线程结束 

    end0 = time.time()
    print("Total time use: %d sec." %(end0-start0))
