import pandas as pd
# i = 1
# for chunk in pd.read_csv('data/train_1.csv',chunksize=2):
#     i+=1
#     print(i,chunk.columns)
#     if(i>2):
#         break
# print(10e4)
import numpy as np

def insert_click_rate(file, rate_file_path, feature_list, output):
    chunksize = 10e5
    iteration = 1
    for chunk in pd.read_csv(file, chunksize=chunksize,usecols=feature_list):
        iteration += 1 
        print('Prosessing {} - {}'.format((iteration-1)*chunksize, iteration * chunksize))
        for column in feature_list: 
              
            rate = pd.read_csv(rate_file_path+'clickVS'+column+'.csv', usecols=[column,'avg(click)'])
            # print('chunk size = ', chunk.shape)
            chunk = pd.merge(chunk,rate,how='left')
            # print('after {} merge ,shape:{}'.format(column,chunk.shape))
            # print('before rename:',chunk.columns)
            chunk.rename(columns={'avg(click)':column+'_rate'}, inplace = True)
            # print('After rename:',chunk.columns)
            # print('drop {}'.format(column))
        chunk.drop(feature_list, inplace = True, axis=1)
        if(iteration%1 == 0):
            print('After drop:',chunk.columns)
        chunk.to_csv(output, index=False, mode='a+')
        
    print('insert rate completed')

train_file = 'data/train_1_time_encode.csv'  
path = 'data/train_info/'
# 先删去DeviceIP，不做考虑
feature_list = ['C1', 'banner_pos', 'site_id', 'site_domain',
       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
    'device_model', 'device_type', 'device_conn_type', 'C14',
       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
# insert_click_rate(file = train_file, 
#     rate_file_path = path, feature_list = feature_list, output = 'data/train_1_rate_test.csv')

# date_parser = lambda x: pd.datetime.strptime(x, '%y%m%d%H')
# def derive_time_features(file, output, start_hour=None, remove_original_feature=False,):
#     for df in pd.read_csv(file, chunksize=10**4, date_parser=date_parser):
#         if start_hour is None:
#             start_hour = df['hour'][0]
            
#         df['hour_int'] = df['hour'].apply(lambda x: np.floor((x - start_hour) / np.timedelta64(1,'h')).astype(np.uint8))
#         df['day_week'] = df['hour'].apply(lambda x: x.dayofweek)
#         df['hour_day'] = df['hour'].apply(lambda x: x.hour)
        
#         if remove_original_feature:
#             df.drop('hour', axis=1, inplace=True)
#         df.to_csv(output, index=False, mode='a+')
# file = '/Users/feixi/Documents/Study/CSDN/Projects/CTR/data/train_1.csv'
# output = 'data/train_1_time_encode_test.csv'
# derive_time_features(file=file,remove_original_feature=True, output=output)

def date_to_hour(df):
    df['hour']=df['hour'].apply(lambda x: x%100)
    return df

df = pd.read_csv('/Users/feixi/Documents/Study/CSDN/Projects/CTR/data/train_1.csv',nrows=10000)
df_2 = date_to_hour(df)
print(df_2.hour)