# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 21:13:53 2018

@author: leikun
"""

import numpy as np
import pandas as pd
import pickle
import os
import time

featrue_version = 'v1_'
train_path = '../Input/train'

# fi = train_path + '/train_sample200w.csv'
# fi = train_path + '/validate_sample10w.csv'
# fi = train_path + '/test_sample10w.csv'

fi = '../Input/test' + '/test.csv' # 全集

# fo = train_path + '/encode/'+featrue_version+'train_sample200wFFM.txt'
# fo = train_path + '/encode/'+featrue_version+'validate_sample10wFFM.txt'
# fo = train_path + '/encode/'+featrue_version+'test_sample10wFFM.txt'

fo = train_path + '/encode/'+featrue_version+'test_fullFFM.txt'

#对于取值很多的特征，计数小于thresholdcout都归于rare类别
thresholdcout = 10

start_hour = pd.datetime.strptime('14102100', '%y%m%d%H')
End_hour = pd.datetime.strptime('14103123', '%y%m%d%H')

min_Hour = (start_hour - start_hour).days * 24 + (start_hour - start_hour).seconds/60/60
max_Hour = (End_hour - start_hour).days * 24 + (End_hour - start_hour).seconds/60/60

field = 'hour'
data_types = {
    'hour': np.str,
    'count(click)': np.uint16,
    'sum(click)': np.uint16,
    'avg(click)': np.float
}
field_sets=pd.read_csv(train_path+'/featurescomb/'+field+'.csv', dtype=data_types,usecols=[field,'avg(click)'])
hour2ctr=dict(zip(list(field_sets[field]), list(field_sets['avg(click)'])))

field = 'device_id'
field_sets=pd.read_csv(train_path+'/featurescomb/'+field+'.csv', usecols=[field,'avg(click)'])
device_id2ctr=dict(zip(list(field_sets[field]), list(field_sets['avg(click)'])))

field = 'device_ip'
field_sets=pd.read_csv(train_path+'/featurescomb/'+field+'.csv', usecols=[field,'avg(click)'])
device_ip2ctr=dict(zip(list(field_sets[field]), list(field_sets['avg(click)'])))

# 对特征取值中计数大于10的进行独热编码,小于10次的统统归为一起,独热编码
count_category_features = ['app_domain',
                        'device_model',
                        'C17', 'C20'
                        ]

# 对特征的每个取值进行独热编码
each_category_features = ['C1','banner_pos',
                        'site_category',
                        'app_category',
                        'device_type','device_conn_type',
                        'C15', 'C16', 'C18', 'C19','C21'
                        ]

# 连续值,对取值不进行独热编码,而是填入数值
continuous_value_features = ['hour',#距离第一天的小时数,最后采用最大最小值归一化填数
                            'device_id','device_ip' #填入点击率,测试集中点击率等于0
                            ]

features_train = count_category_features+ each_category_features+continuous_value_features

features_test =  count_category_features+ each_category_features+continuous_value_features

date_parser = lambda x: pd.datetime.strptime(x, '%y%m%d%H')
data_types = {
    'id': np.str,
    'click': np.uint16,
    'hour': np.str,
    'C1': np.uint16,
    'banner_pos': np.uint16,
    'site_id': np.object,
    'site_domain': np.object,
    'site_category': np.object,
    'app_id': np.object,
    'app_domain': np.object,
    'app_category': np.object,
    'device_id': np.object,
    'device_ip': np.object,
    'device_model': np.object,
    'device_type': np.uint16,
    'device_conn_type': np.uint16,
    'C14': np.uint16,
    'C15': np.uint16,
    'C16': np.uint16,
    'C17': np.uint16,
    'C18': np.uint16,
    'C19': np.uint16,
    'C20': np.uint16,
    'C21': np.uint16    
}

train = pd.read_csv(fi,
                    dtype=data_types,
                    chunksize=10000)

# loading dicts
features_train_dict = {}
for feature in features_train:
    with open(train_path+'/dicts/'+featrue_version+feature+'.pkl','rb') as f:
        features_train_dict[feature] = pickle.load(f)

# features_test_dict = {}
# for feature in features_test:
#     with open(train_path+'/dicts/'+featrue_version+feature+'.pkl','rb') as f:
#         features_test_dict[feature] = pickle.load(f)

with open(train_path+'/features/'+featrue_version+'feature2field.pkl', 'rb') as f:
    feature2field = pickle.load(f)

# 编码
def encoder2ffm(data_line, features_dict, fields_dict):
    """
    将每行数据文件进行编码处理后，返回libffm的格式

    :data_line：训练集或者测试集的每行数据，为list变量

    :features_dict: 是所有特征的编号编码字典，格式为：
    {...，特征i：{值1：列号1，值2：列号2，...}，...}，编号值相当于编码后的每一列编号，label列为0，往后依次为1，2，3，...

    :fields_dict: 编码后的特征的域编码字典，格式为：
    {...，特征i_值1:域号，特征i_值2:域号，，特征i_值k:域号，...}，对于类别型的特征，<值k>就是所有类别的取值；
    对于根据计数进行划归类别的，需要注意计数小于thresholdcout的取值都当作rare类别

    :return: 经过编码的libffm格式行数据，格式为：
    标签 域i编号:特征j编号:特征j的取值 域i编号:特征j+1编号:特征j+1的取值 .... 域i+1编号:特征j+m编号:特征j+m的取值
    域i和特征j的编号是必须要递增的。

    """
    if 'click' in data_line:
        ffmline = str(data_line['click']) # 训练和验证数据集,首先是标签值
    else:
        ffmline ='0' # 测试数据集没有标签,但是需要添加一列label当作占位符，否则会把第一列当作label处理了。
    

    # 对每个特征进行遍历,处理
    # features_dict的keys=[... 'C17', 'C20', 'C1',...,'C19', 'C21', 'hour', 'device_id', 'device_ip']

    for feature in features_dict:
        # 当前行,特征field的取值
        feature_value = data_line[feature]

        # 连续变量需要填入的不是0/1,而是归一化后的数值
        if feature =='hour':
            encode_value = hour2ctr[feature_value]
            feature_value = 'conti_value'
            feature_index = features_dict[feature][feature_value]

        elif feature =='device_id':
            encode_value = device_id2ctr[feature_value]
            feature_value = 'conti_value'
            feature_index = features_dict[feature][feature_value]

        elif feature =='device_ip':
            if not device_ip2ctr.__contains__(feature_value):
                print(data_line)

            encode_value = device_ip2ctr[feature_value]
            feature_value = 'conti_value'
            feature_index = features_dict[feature][feature_value]

        else:
            encode_value = 1
            if features_dict[feature].__contains__(feature_value):
                feature_index = features_dict[feature][feature_value]
            else:
                feature_value = 'rare'
                feature_index = features_dict[feature][feature_value]

        # 获得域编码        
        field_index = fields_dict[feature+'_'+str(feature_value)]

        ffmline += ' ' + str(field_index) + ':' + str(feature_index) + ':' + str(encode_value)

    # 删除首尾的空格
    ffmline.strip()

    return ffmline

# 添加新特征
def add_features(df, start_hour=None, remove_original_feature=False):
    if start_hour is None:
        start_hour =pd.datetime.strptime('14102100', '%y%m%d%H') # df['hour'][0]
    
    # 添加时间相关的三个特征
    df['hour_int'] = df['hour'].apply(lambda x: np.floor((x - start_hour) / np.timedelta64(1, 'h')).astype(np.uint16))
    df['day_week'] = df['hour'].apply(lambda x: x.dayofweek)
    df['hour_day'] = df['hour'].apply(lambda x: x.hour)
    
    # 添加屏幕面积特征
    df['C15XC16'] = df['C15'].astype(str)+'X'+df['C16'].astype(str)
        
    if remove_original_feature:
        df.drop('hour', axis=1, inplace=True)
    
    return df, start_hour


path = '../Input/train/encode'
if not os.path.exists(path):
    os.makedirs(path)

start0 = time.time()

for j,dc in enumerate(train):

    start = time.time()

    print("chunk"+str(j)+" processing...")
    
    # 对每行进行处理,转化为libffm格式的字符串
    chunk=[]
    for i in range(len(dc)):
        line = dc.iloc[i,:]
        chunk.append(encoder2ffm(line, features_train_dict, feature2field))

    pd.DataFrame(chunk).to_csv(fo,mode='a',index=False,header=False)

    end = time.time()
    print('chunk'+str(j)+' processed,time used: %d sec.' %(end-start))

end0 = time.time()
print("Total time use: %d sec." %(end0-start0))

print("libffm format file ref. to " + fo )
