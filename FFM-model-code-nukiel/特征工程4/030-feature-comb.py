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

# 将将test.csv文件中各个特征取值多出来的值和train中的值合并一起

# 例如特征C21的训练集统计数据文件为:clickVSC21.csv
# 其内容为:
# C21	count(click)	sum(click)	avg(click)
# 1	    3027	        416	        0.1374297985
# 13	384314	        81705	    0.212599593
# 15	766986	        161746	    0.2108852052
# 16	347178	        90093	    0.25950089
# 17	166268	        15100	    0.0908172348
# 20	13365	        1204	    0.0900860456
# 23	8896205	        1894394	    0.2129440587
# 32	1783343	        410781	    0.2303432374
# 33	1497559	        589917	    0.3939190376
# 35	46707	        13286	    0.2844541503
# ...

# count(click)列为每个取值的统计数
# sum(click)列为click=1的统计数
# avg(click)列为点击率,即sum(click)/count(click)

# 而C21在测试集中还有184和240两个取值是不在这个统计文件的,这时候需要把这两个取值放在这个文件里
# 而count,sum,avg的取值直接赋值0

# 合并以后另存到./featurescomb/C21.csv

#import numpy as np
import pandas as pd
import os

features = ['hour', 'C1', 'banner_pos',
          'site_id', 'site_domain', 'site_category',
          'app_id', 'app_domain', 'app_category',
          'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type',
          'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

train_path="/media/leikun/programs/AI100/AI100-Final/Input/train"
test_path="/media/leikun/programs/AI100/AI100-Final/Input/test"

path ="/media/leikun/programs/AI100/AI100-Final/Input/train/featurescomb"
if not os.path.exists(path):
    os.makedirs(path)

for field in features:

    # 输出的文件
    fo = '../Input/train/featurescomb/'+ field + '.csv'

    # 读入训练集特征的统计表
    f1 = train_path+'/features/clickVS'+field+'.csv' 
    train_sets=pd.read_csv(f1)

    # 读入测试集中特征field多出的取值
    test_sets=[]
    f2 = test_path+'/onlytest/test_more_cat_'+field+'.txt'
    if os.path.getsize(f2):
        test_sets=pd.read_csv(f2,header=None)

    # 如果有值才进行合并
    if len(test_sets) > 0 :
        add_rows = []
        for v in list(test_sets[0]):
            add_rows.append([v,0,0,0])

        add_pd=pd.DataFrame(add_rows)
        add_pd.columns = train_sets.columns
        train_sets = pd.concat([train_sets,add_pd],ignore_index=True)

    # 将合并的数据保存
    train_sets.to_csv(fo,mode='w',index=False)

#####################################

features = ['ad_id', 'user_id']

for field in features:

    # 输出的文件
    fo = path +'/'+ field + '.csv'

    # 读入训练集特征的统计表
    f1 = train_path+'/trainhour/summary/'+field+'.csv' 
    train_sets=pd.read_csv(f1,usecols=[field])

    # 读入测试集中特征的统计表
    test_sets=[]
    f2 = test_path+'/testhour/summary/'+field+'.csv'
    test_sets=pd.read_csv(f2,usecols=[field])

    sum_sets = pd.DataFrame(list(set(list(train_sets[field])+list(test_sets[field]))),columns=[field])

    # 将合并的数据保存
    sum_sets.to_csv(fo,mode='w',index=False)

