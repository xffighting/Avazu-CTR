# 将统计数据按fields处理
# 例如特征banner_pos的统计数据文件为:clickVSbanner_pos.csv
# 其内容为:

# banner_pos	count(click)	sum(click)	avg(click)
# 0	            29109590	    4781901	    0.1642723584
# 1	            11247282	    2065164	    0.1836144946
# 2	            13001	        1550	    0.1192215983
# 3	            2035	        372	        0.1828009828
# 4	            7704	        1428	    0.1853582555
# 5	            5778	        702	        0.1214953271
# 7	            43577	        13949	    0.3201000528

# banner_pos列为该特征在训练集中的所有取值
# count(click)列为每个取值的统计数
# sum(click)列为click=1的统计数
# avg(click)列为点击率,即sum(click)/count(click)

# 对于直接编码的特征,将提取第一列(banner_pos),即为一个field,每个取值归为一个域,生成一个特征字典文件
# 对于按频率编码的特征,将提取第一列和第二列,小于10次的取值归于一个域,
"""
| 特征              | 特征取值                                        | 特征取值总数   |
| ---------------- | ---------------------------------------------- | ------------ |
| id               | 5.21159374e+11                                 | 40428967     |
| click            | 0,1                                            | 2            |
| hour             | 14102100 14102101                              | 240          |
| C1               | 1001 1002 1005 1007 1008 1010 1012             | 7            |
| banner_pos       | 0 1 2 3 4 5 7                                  | 8            |
| site_id          | '000aa1a4' ' ... 'fffe8e1c'                    | 4737         |
| site_domain      | '004d30ed' ... 'ffdec903'                      | 7745         |
| site_category    | '0569f928' …'f66779e6'                         | 26           |
| app_id           | '000d6291' ... 'ffef3b38'                      | 8552         |
| app_domain       | 'fea0d84a'…'ff6630e0'                          | 559          |
| app_category     | 'd1327cf5' …'fc6fa53d'                         | 36           |
| device_id        | '00000919' ... 'ffffde2c'                      | 2686408      |
| device_ip        | '00000911' ... 'fffff971'                      | 6729486      |
| device_model     | '000ab70c' ... 'ffe72be2'                      | 8251         |
| device_type­­­      | 0 1 2 4 5                                      | 5            |
| device_conn_type | 0 2 3 5                                        | 4            |
| C14              | 375 ... 24052                                  | 2626         |
| C15              | 120    216  300  320    480  728  768 1024     | 8            |
| C16              | 20     36   50   90    250  320  480  768 1024 | 9            |
| C17              | 112…2758                                       | 435          |
| C18              | 0 1 2 3                                        | 4            |
| C19              | 33…1959                                        | 68           |
| C20              | -1 100000 100001…100248                        | 172          |
| C21              | 1 … 219                                        | 60           |

"""

#import numpy as np
import pandas as pd
import pickle
import itertools

# 记录不同的特征工程

featrue_version='v2_' # 不考虑增加特征

# 对特征取值中计数大于10的进行独热编码,小于10次的统统归为一起,独热编码
count_category_features = ['site_id','site_domain',
                        'app_id','app_domain',
                        'device_model',
                        'C14','C17', 'C20'
                        ]

# 对特征的每个取值进行独热编码
each_category_features = ['C1','banner_pos',
                        'site_category',
                        'app_category',
                        'device_type','device_conn_type',
                        'C15', 'C16', 'C18', 'C19','C21'
                        ]

# 连续值,对取值不进行独热编码,而是填入数值
continuous_value_features = ['hour',#填入点击率
                            'device_id','device_ip' #填入点击率,测试集中点击率等于0
                            ]

# 每个特征的fields编号
fields_index={}

fields_index['site_id'] = 1
fields_index['site_domain'] = 2
fields_index['app_id'] = 3
fields_index['app_domain'] = 4
fields_index['device_model'] = 5
fields_index['C14'] = 6
fields_index['C17'] = 6
fields_index['C20'] = 7

fields_index['C1'] = 8
fields_index['banner_pos'] = 9
fields_index['site_category'] = 10
fields_index['app_category'] = 11
fields_index['device_type'] = 12
fields_index['device_conn_type'] = 13
fields_index['C15'] = 14
fields_index['C16'] = 14
fields_index['C18'] = 15
fields_index['C19'] = 16
fields_index['C21'] = 17

fields_index['hour'] = 18
fields_index['device_id'] = 19
fields_index['device_ip'] = 19

train_path='../Input/train'

feature2field = {}
ind = 1

for field in count_category_features:
    field_dict = {}
    #载入第一列和第二列,并组合为一个字典
    field_sets=pd.read_csv(train_path+'/featurescomb/'+field+'.csv', usecols=[field,'count(click)'])
    field2count=dict(zip(list(field_sets[field]), list(field_sets['count(click)'])))
    index_rare = None
    for value,count in field2count.items():
        if count < 10:
            if index_rare == None:
                field_dict['rare'] = ind
                feature2field[field+'_'+str('rare')] = fields_index[field]
                index_rare = ind
                ind += 1
            else:
                field_dict['rare'] = index_rare
                feature2field[field+'_'+str('rare')] = fields_index[field]
        else:
            field_dict[str(value)] = ind
            feature2field[field+'_'+str(value)] =  fields_index[field]
            ind += 1

    with open(train_path+'/dicts/'+featrue_version+field+'.pkl', 'wb') as f:
        pickle.dump(field_dict, f)


for field in each_category_features:
    # value to one-hot-encoding index dict
    field_dict = {}

    if field=='hour_day':
        field_sets = pd.DataFrame({'hour_day':list(range(24))})
    elif field=='day_week':
        field_sets = pd.DataFrame({'day_week':list(range(7))})
    elif field=='C15XC16':
        C15=pd.read_csv(train_path+'/featurescomb/'+'C15'+'.csv', usecols=['C15'])
        C16=pd.read_csv(train_path+'/featurescomb/'+'C16'+'.csv', usecols=['C16'])
        C1516=list(itertools.product(list(C15['C15']),list(C16['C16'])))

        field_sets = pd.DataFrame({'C15XC16':list(str(x[0])+'X'+str(x[1]) for x in C1516)})
    else:
        field_sets=pd.read_csv(train_path+'/featurescomb/'+field+'.csv', usecols=[field])

    for value in list(field_sets[field]):
        field_dict[str(value)] = ind
        feature2field[field+'_'+str(value)] = fields_index[field]
        ind += 1

    with open(train_path+'/dicts/'+featrue_version+field+'.pkl', 'wb') as f:
        pickle.dump(field_dict, f)


for field in continuous_value_features:
    # 连续变量
    field_dict = {}
    value = 'conti_value'
    field_dict[value] = ind
    feature2field[field+'_'+str(value)] = fields_index[field]
    ind += 1

    with open(train_path+'/dicts/'+featrue_version+field+'.pkl', 'wb') as f:
        pickle.dump(field_dict, f)


with open(train_path+'/features/'+featrue_version+'feature2field.pkl', 'wb') as f:
    pickle.dump(feature2field, f)


# 输出CSV 是为了检查
feature2field_csv=pd.DataFrame(list(feature2field.values()),list(feature2field))
feature2field_csv.to_csv(train_path+'/features/'+featrue_version+'feature2field.csv',mode='w')
