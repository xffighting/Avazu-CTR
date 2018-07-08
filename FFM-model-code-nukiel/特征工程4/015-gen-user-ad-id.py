import random
import pandas as pd
import os
import time
import threadpool
import hashlib

def hashstr(str, nr_bins=1e6):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1



start0 = time.time()


f1 = "/media/leikun/programs/AI100/AI100-Final/Input/train/featurescomb/device_ip.csv"
f2 = "/media/leikun/programs/AI100/AI100-Final/Input/train/featurescomb/device_model.csv"

f3 = "/media/leikun/programs/AI100/AI100-Final/Input/train/featurescomb/C14.csv"
f4 = "/media/leikun/programs/AI100/AI100-Final/Input/train/featurescomb/C17.csv"
f5 = "/media/leikun/programs/AI100/AI100-Final/Input/train/featurescomb/banner_pos.csv"


df1 = pd.read_csv(f1,usecols=['device_ip'])
df2 = pd.read_csv(f2,usecols=['device_model'])

user_id=[]
for x in list(df1['device_ip']):
    for y in list(df2['device_model']):
        user_id.append(hashstr(x+'+'+y))

user_id = pd.DataFrame(list(set(user_id)),columns=['user_id'])

# 数据保存
fo = "/media/leikun/programs/AI100/AI100-Final/Input/train/featurescomb/user_id.csv"
user_id.to_csv(fo,mode='w',index=False)

df1 = pd.read_csv(f3,usecols=['C14'])
df2 = pd.read_csv(f4,usecols=['C17'])
df3 = pd.read_csv(f5,usecols=['C17'])

ad_id=[]
for x in list(df1['C14']):
    for y in list(df2['C17']):
        for z in list(df3['C17']):
            ad_id.append(hashstr(str(x)+'+'+str(y)+'+'+str(z)))

ad_id = pd.DataFrame(list(set(ad_id)),columns=['ad_id'])
 
fo = "/media/leikun/programs/AI100/AI100-Final/Input/train/featurescomb/user_id.csv"
ad_id.to_csv(fo,mode='w',index=False)

end0 = time.time()
print("Total time use: %d sec." %(end0-start0))