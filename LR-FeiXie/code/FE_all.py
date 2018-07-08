import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV


#保留的特征：目前去掉了Hour， 因为下面对hour重新进行的了特征提取
#去掉device_ip
feature_list = ['C1', 'banner_pos', 'site_id', 'site_domain',
       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
       'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
use_cols = ['hour','C1', 'banner_pos', 'site_id', 'site_domain',
       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
       'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']


def insert_click_rate(file, rate_file_path, feature_list, use_cols, output):
    chunksize = 10e5
    iteration = 1
    for chunk in pd.read_csv(file, chunksize=chunksize, usecols=use_cols):
        iteration += 1
        chunk['hour']=chunk['hour'].apply(lambda x: x % 100 / 24) 
        print('Prosessing {} - {}'.format((iteration-1)*chunksize, iteration * chunksize))
        for column in feature_list: 
            rate = pd.read_csv(rate_file_path+'clickVS'+column+'.csv', usecols=[column,'avg(click)'])
            chunk = pd.merge(chunk,rate,how='left')
            chunk.rename(columns={'avg(click)':column+'_rate'}, inplace = True)
        chunk.drop(feature_list, inplace = True, axis=1)
        if(iteration%100 == 0):
            print('After drop:',chunk.columns)
        chunk.to_csv(output, index=False, mode='a+')
    print('insert rate completed')


# f = "../Input/train/train_sample10w.csv"
# f = "/Users/feixi/Documents/Study/CSDN/Projects/CTR/data/test"
f = "/Users/feixi/Documents/Study/CSDN/Projects/CTR/data/train_sample50W.csv"
# train_df = load_train_data(f)
# derive_time_features(train_df, remove_original_feature=True)

# print(train_df.head())
# print(train_df.shape)
# train_df.to_csv('data/train_50w_time_encode.csv',index=False)

train_file = 'data/train_1.csv'  
path = 'data/train_info/'
# 先删去DeviceIP，不做考虑

insert_click_rate(file = train_file, 
    rate_file_path = path, feature_list = feature_list, output = 'data/train_1_rate_test.csv', use_cols=use_cols)
