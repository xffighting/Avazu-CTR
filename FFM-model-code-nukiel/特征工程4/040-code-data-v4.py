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
import datetime
import hashlib
import shutil
import threading
import threadpool
from decimal import Decimal

featrue_version = 'v4' #

train_path ="/media/leikun/programs/AI100/AI100-Final/Input/train"
test_path ="/media/leikun/programs/AI100/AI100-Final/Input/test"

TotalFiles =0
 
# 对特征的每个取值进行独热编码
each_category_features = ['C1','banner_pos',
                        'site_category',
                        'app_category',
                        'device_type','device_conn_type',
                        'C15', 'C16', 'C15XC16','C18', 'C19','C20','C21',
                        'day_week','hour_day'
                        ]

# #距离第一天的小时数,采用最大最小值归一化填数
continuous_value_features = ['hour_int']

# 对特征的每个取值衍生出当前小时出现次数x_times,前一个小时出现次数想_timespre,前后小数出现次数增量x_timesrate
# 对于衍生的每个取值的三个特征都采用最大最小值归一化
derive_category_features = ['C1_times','banner_pos_times',
                        'site_category_times',
                        'app_category_times',
                        'app_domain_times',
                        'C15_times', 'C16_times', 'C18_times', 'C19_times','C20_times','C21_times',
                        'ad_id_times']

# ad_id : hashstr(C14+C17+banner_pos)
# user_id : hashstr(device_ip+device_model)

# 对广告ad_id对其他在特征衍生出的当前小时出现次数x_times,前一个小时出现次数想_timespre,前后小数出现次数增量x_timesrate
# 对于衍生的这三个特征为连续变量,采用最大最小值归一化
derive_ad_features = ['ad_f_user_times']

# ad_f_user :ad_id对user_id展示的次数
# ad_f_user_times: 为当前小时展示的次数
# ad_f_user_timespre: 为之前一小时展示的次数
# ad_f_user_timesrate: 为前后小时展示次数的增量


features_train =  each_category_features+continuous_value_features +derive_category_features #+derive_ad_features


date_parser = lambda x: pd.datetime.strptime(x, '%y%m%d%H')

data_types = {
    'id': np.str,
    'click': np.uint16,
    'hour': np.str  
}


fis = [ train_path + '/mini_trainv4.csv',
train_path + '/validate_sample10w.csv',
train_path + '/test_sample10w.csv',
test_path + '/test.csv']

# fi = train_path + '/train_sample200w.csv'
# fi = train_path + '/validate_sample10w.csv'
# fi = train_path + '/test_sample10w.csv'
# fi = '../Input/test' + '/test.csv' # 全集

# fo = train_path + '/encode/'+featrue_version+ "_"+'train_sample200wFFM.txt'
# fo = train_path + '/encode/'+featrue_version+ "_"+'validate_sample10wFFM.txt'
# fo = train_path + '/encode/'+featrue_version+ "_"+'test_sample10wFFM.txt'
# fo = train_path + '/encode/'+featrue_version+ "_"+'test_fullFFM.txt'

out_path = "/media/leikun/programs/AI100/AI100-Final/Input/train/" + featrue_version+ "_" + "data"
if not os.path.exists(out_path):
    os.makedirs(out_path)

fos1 = [ out_path + '/mini_train'+featrue_version+'.ffm',
out_path + '/validate_sample10w'+featrue_version+'.ffm',
out_path + '/test_sample10w'+featrue_version+'.ffm',
out_path + '/test'+featrue_version+'.ffm']

fos2 = [ out_path + '/mini_train'+featrue_version+'.fm',
out_path + '/validate_sample10w'+featrue_version+'.fm',
out_path + '/test_sample10w'+featrue_version+'.fm',
out_path + '/test'+featrue_version+'.fm']


min_max1 = {'hour_int': {'minc': 0.0, 'maxc': 263.0, 'minr': 1, 'maxr': 1}, 
            'ad_f_user': {'minc': 0, 'maxc': 1846, 'minr': -1846.0, 'maxr': 1846.0}}
min_max2 =  {'hour_int': {'minc': 0.0, 'maxc': 263.0, 'minr': 1, 'maxr': 1},
            'ad_f_user': {'minc': 0, 'maxc': 3418, 'minr': -3418.0, 'maxr': 3418.0}}

################### 汇总统计次数到dataframe-训练集

times_features = ['banner_pos',
                        'site_category',
                        'app_category',
                        'app_domain',
                        'C15', 'C16', 'C18', 'C19','C20','C21',
                        'ad_id']

# 第一个特征
myfield = 'C1'
fea_times1 = pd.read_csv(train_path+'/trainhour/summary/'+myfield+'.csv')
fea_times1['fea'] = fea_times1[myfield].apply(lambda x: myfield + "_" +str(x))
fea_times1.drop(myfield, axis=1, inplace=True)

# 将所有特征按照行合并
for myfield in times_features:
    fea_times=pd.read_csv(train_path+'/trainhour/summary/'+myfield+'.csv')
    fea_times['fea'] = fea_times[myfield].apply(lambda x: myfield + "_" +str(x))
    fea_times.drop(myfield, axis=1, inplace=True)
    fea_times1 = pd.concat([fea_times1,fea_times])

# 将特征_值列作为行索引
fea_times1 = fea_times1.set_index('fea')

# 增加14102023这个小时,赋值为均值
fea_times1['14102023'] = list(fea_times1.mean(axis=1))
fea_times1['14102023'] = fea_times1['14102023'].astype('int')

# 把各个列按照时间先后排序
day=['141021','141022','141023','141024','141025','141026','141027','141028','141029','141030']
hour=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
param_list=['14102023']
for d in day:
    for h in hour:
        param_list.append(d+h)
            
fea_times1 = fea_times1[param_list]

# 保存最后一天的23小时,作为测试集的第一天0时的前一个小数数据
Test_col = fea_times1['14103023']

# 求得前后两列的差值,即次数的增加率
fea_rate1=fea_times1.diff(axis=1)

# 第一列是没有差值的,直接补0
fea_rate1 = fea_rate1.fillna(0)

# 装置,便于归一化
fea_times1 = fea_times1.T
fea_rate1 = fea_rate1.T

# 最大最小归一化
fea_times1 = (fea_times1 - fea_times1.min()) / (fea_times1.max() - fea_times1.min())
fea_rate1 = (fea_rate1 - fea_rate1.min()) / (fea_rate1.max() - fea_rate1.min())

# 某时刻h某个特征_值v的次数(归一化)为:
# fea_times1[v][h]

###################

################### 汇总统计次数到dataframe-测试集


times_features = ['banner_pos',
                        'site_category',
                        'app_category',
                        'app_domain',
                        'C15', 'C16', 'C18', 'C19','C20','C21',
                        'ad_id']

# 第一个特征
myfield = 'C1'
fea_times2 = pd.read_csv(test_path+'/testhour/summary/'+myfield+'.csv')
fea_times2['fea'] = fea_times2[myfield].apply(lambda x: myfield + "_" +str(x))
fea_times2.drop(myfield, axis=1, inplace=True)

# 将所有特征按照行合并
for myfield in times_features:
    fea_times=pd.read_csv(test_path+'/testhour/summary/'+myfield+'.csv')
    fea_times['fea'] = fea_times[myfield].apply(lambda x: myfield + "_" +str(x))
    fea_times.drop(myfield, axis=1, inplace=True)
    fea_times2 = pd.concat([fea_times2,fea_times])

# 将特征_值列作为行索引
fea_times2 = fea_times2.set_index('fea')

# 增加14103023这个小时,赋值为均值
fea_times2['14103023'] = list(fea_times2.mean(axis=1))
fea_times2['14103023'] = fea_times2['14103023'].astype('int')

# 把各个列按照时间先后排序
day=['141031']
hour=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
param_list=['14103023']
for d in day:
    for h in hour:
        param_list.append(d+h)
            
fea_times2 = fea_times2[param_list]

# 训练集中有的值,就拿过来
vals1 = list(Test_col.index)
vals2 = list(fea_times2.index) 

for i in vals2:
    if vals1.__contains__(i):
        fea_times2['14103023'][i] = Test_col[i]
            
# 求得前后两列的差值,即次数的增加率
fea_rate2=fea_times2.diff(axis=1)

# 第一列是没有差值的,直接补0
fea_rate2 = fea_rate2.fillna(0)

# 装置,便于归一化
fea_times2 = fea_times2.T
fea_rate2 = fea_rate2.T

# 最大最小归一化
fea_times2 = (fea_times2 - fea_times2.min()) / (fea_times2.max() - fea_times2.min())
fea_rate2 = (fea_rate2 - fea_rate2.min()) / (fea_rate2.max() - fea_rate2.min())

# 某时刻h某个特征_值v的次数(归一化)为:
# fea_times2[v][h]

###################


# 每一列特征的编号,即列号
features_train_dict = {}
for feature in features_train:
    with open(train_path+'/dicts/'+featrue_version+ "_"+feature+'.pkl','rb') as f:
        features_train_dict[feature] = pickle.load(f)
# features_train_dict是每个feature不同取值的编号(列号)
# features_train_dict['C15_times']
# {'1024_times': 2506,
#  '1024_timespre': 2507,
# .....
#  '728_timesrate': 2502,
#  '768_times': 2503,
#  '768_timespre': 2504,
#  '768_timesrate': 2505}


# 每一列特征的归属域编号
with open(train_path+'/features/'+featrue_version+ "_"+'feature2field.pkl', 'rb') as f:
    feature2field = pickle.load(f)
# feature2field是每个列标题的归属域编号
# feature2field['C15_1024_times']


def hashstr(str, nr_bins=1e6):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1


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
        fmline = str(data_line['click'])
    else:
        ffmline = '0' # 测试数据集没有标签,但是需要添加一列label当作占位符，否则会把第一列当作label处理了。
        fmline = '0'
    

    # 对每个特征进行遍历,处理
    # features_dict的keys=[... 'C14', 'C17', 'C20', 'C1',
    #  'banner_pos', 'site_category', 'app_category', 'device_type',
    #  'device_conn_type', 'C15', 'C16', ...]

    for feature in features_dict:
        # 当前行,特征field的取值

        fea = feature
        if '_times' in feature:
            fea = feature.replace('_times','') # 去掉_times后缀
        
        # 连续变量需要填入的不是0/1,而是归一化后的数值
        if feature in continuous_value_features:

            if feature == 'hour_int':
                start_hour = pd.datetime.strptime('14102100', '%y%m%d%H')
                mydate = pd.datetime.strptime(data_line['hour'], '%y%m%d%H')
                feature_value = (mydate-start_hour).total_seconds()/60/60
            else:
                feature_value = data_line[fea] # 去原始数据中提取值,是必须没有_times后缀的

            vmax = 0
            vmin = 0
            if min_max1.__contains__(feature):
                vmax = min_max1[feature]['maxc']
                vmin = min_max1[feature]['minc']
            
            if min_max2.__contains__(feature):
                vmax = max(vmax,min_max2[feature]['maxc'])
                vmin = min(vmin,min_max2[feature]['minc'])

            if vmax-vmin==0:
                encode_value = 0 #如果出现这个情况就只能直接先给个0了
                print('Error: max and min of ' + feature + ' is the same!!!!!' )
            else:
                encode_value = (feature_value-vmin) /(vmax-vmin)

            feature_value = 'conti_value'
            if features_dict.__contains__(feature):
                if features_dict[feature].__contains__(feature_value):
                    feature_index = features_dict[feature][feature_value]
                else:
                    feature_index = -999
                    print('Error:features_dict[' + feature + '] has no key of ' + str(feature_value))
            else:
                feature_index = -999
                print('Error:features_dict has no key of ' + feature )

            # 获得域编码        
            field_index = fields_dict[feature+'_'+str(feature_value)]

            if encode_value < 0.000001:
                encode_value =0
            elif encode_value ==1:
                encode_value = 1
            else:
                encode_value =Decimal(encode_value).quantize(Decimal('0.000000'))

            if feature_index != -999 :
                ffmline += ' ' + str(field_index) + ':' + str(feature_index) + ':' + str(encode_value)
                fmline += ' ' + str(feature_index) + ':' + str(encode_value)

        # 独热编码特征
        if feature in each_category_features:
            encode_value = 1

            if feature == 'C15XC16':
                feature_value = data_line['C15'].astype(str)+'X'+data_line['C16'].astype(str)
            elif feature == 'day_week':
                mydate = pd.datetime.strptime(data_line['hour'], '%y%m%d%H')
                feature_value = mydate.weekday()
            elif feature == 'hour_day':
                mydate = pd.datetime.strptime(data_line['hour'], '%y%m%d%H')
                feature_value = mydate.hour
            else:
                feature_value = data_line[fea] # 去原始数据中提取值,是必须没有_times后缀的

            if features_dict.__contains__(feature):
                if features_dict[feature].__contains__(feature_value):
                    feature_index = features_dict[feature][feature_value]
                else:
                    feature_index = -999
                    print('Error:features_dict[' + feature + '] has no key of ' + str(feature_value))
            else:
                feature_index = -999
                print('Error:features_dict has no key of ' + feature )

            # 获得域编码        
            field_index = fields_dict[feature+'_'+str(feature_value)]

            if feature_index != -999 :
                ffmline += ' ' + str(field_index) + ':' + str(feature_index) + ':' + str(encode_value)
                fmline += ' ' + str(feature_index) + ':' + str(encode_value)

        # 特征按照时间统计变量
        if feature in derive_category_features:
            
            myhour = data_line['hour']
            
            if fea == 'ad_id':
                myvalue = int(hashstr(str(data_line['C14'])+'+'+str(data_line['C17'])+'+'+str(data_line['C17'])))
                feature_value = myvalue
            else:
                feature_value = data_line[fea]
                myvalue = feature_value

            hournow =  pd.datetime.strptime(myhour, '%y%m%d%H')
            myhourpre = hournow - datetime.timedelta(hours = 1)
            myhourpre = myhourpre.strftime('%y%m%d%H')

            if '141031' in str(myhour): # 测试集数据
                try:
                    times = fea_times2[fea+"_"+str(myvalue)][str(myhour)]
                    timespre = fea_times2[fea+"_"+str(myvalue)][str(myhourpre)]
                    timesrate = fea_rate2[fea+"_"+str(myvalue)][str(myhour)]
                except:
                    print('Error: fea_times2[' + fea+'_'+str(myvalue) + ']['+str(myhour)+']' )
                    times = 0
                    timespre = 0
                    timesrate = 0
            else:
                # 训练集
                try:
                    times = fea_times1[fea+"_"+str(myvalue)][str(myhour)]
                    timespre = fea_times1[fea+"_"+str(myvalue)][str(myhourpre)]
                    timesrate = fea_rate1[fea+"_"+str(myvalue)][str(myhour)]
                except:
                    print('Error: fea_times2[' + fea+'_'+str(myvalue) + ']['+str(myhour)+']' )
                    times = 0
                    timespre = 0
                    timesrate = 0

            if times < 0.000001:
                times =0
            elif times ==1:
                times = 1
            else:
                times =Decimal(times).quantize(Decimal('0.000000'))

            if timespre < 0.000001:
                timespre =0
            elif timespre ==1:
                timespre = 1
            else:
                timespre =Decimal(timespre).quantize(Decimal('0.000000'))

            if timesrate < 0.000001:
                timesrate =0
            elif timesrate ==1:
                timesrate = 1
            else:
                timesrate =Decimal(timesrate).quantize(Decimal('0.000000'))

            ##################################################
            myfeature_value = str(myvalue) + '_times'
            if features_dict.__contains__(feature):
                if features_dict[feature].__contains__(myfeature_value):
                    feature_index = features_dict[feature][myfeature_value]
                else:
                    feature_index = -999
                    print('Error:features_dict[' + feature + '] has no key [' + str(myfeature_value) +']' )
            else:
                feature_index = -999
                print('Error:features_dict has no key of ' + feature )

            # 获得域编码,特征名不带_times后缀,用fea这个处理后的        
            field_index = fields_dict[fea+'_'+myfeature_value]

            if feature_index != -999 :
                ffmline += ' ' + str(field_index) + ':' + str(feature_index) + ':' + str(times)
                fmline += ' ' + str(feature_index) + ':' + str(times)

            ##################################################
            myfeature_value = str(myvalue) + '_timespre'
            if features_dict.__contains__(feature):
                if features_dict[feature].__contains__(myfeature_value):
                    feature_index = features_dict[feature][myfeature_value]
                else:
                    feature_index = -999
                    print('Error:features_dict[' + feature + '] has no key of ' + str(myfeature_value)+']' )
            else:
                feature_index = -999
                print('Error:features_dict has no key of ' + feature )

            # 获得域编码,特征名不带_times后缀,用fea这个处理后的        
            field_index = fields_dict[fea+'_'+myfeature_value]

            if feature_index != -999 :
                ffmline += ' ' + str(field_index) + ':' + str(feature_index) + ':' + str(timespre)
                fmline += ' ' + str(feature_index) + ':' + str(timespre)

            ##################################################
            myfeature_value = str(myvalue) + '_timesrate'
            if features_dict.__contains__(feature):
                if features_dict[feature].__contains__(myfeature_value):
                    feature_index = features_dict[feature][myfeature_value]
                else:
                    feature_index = -999
                    print('Error:features_dict[' + feature + '] has no key of ' + str(myfeature_value) +']' )
            else:
                feature_index = -999
                print('Error:features_dict has no key of ' + feature )

            # 获得域编码,特征名不带_times后缀,用fea这个处理后的        
            field_index = fields_dict[fea+'_'+myfeature_value]

            if feature_index != -999 :
                ffmline += ' ' + str(field_index) + ':' + str(feature_index) + ':' + str(timesrate)
                fmline += ' ' + str(feature_index) + ':' + str(timesrate)

    return ffmline,fmline
  

def split_files():

    for j,fi in enumerate(fis):

        train = pd.read_csv(fi,
                        dtype=data_types,
                        chunksize=20000)

        out_path_temp = "/media/leikun/programs/AI100/AI100-Final/Input/train/" + featrue_version+ "_" + "data/temp" + str(j)
        if not os.path.exists(out_path_temp):
            os.makedirs(out_path_temp)
        else:
            shutil.rmtree(out_path_temp)
            os.makedirs(out_path_temp)

        for k,dc in enumerate(train):

            dc.to_csv(out_path_temp+'/'+str(k)+'.csv',mode='w',index=False,header=True)

            print(os.path.basename(fi) + ': chunk'+str(k)+' processed')

        print(os.path.basename(fi) + ' processed')

def make_model_data(fi,ffmo,fmo):

    global  TotalFiles
    TotalFiles = TotalFiles -1
    print(os.path.basename(ffmo) + "....." + " of " + os.path.basename(fi)+ ' Left Files:' + str(TotalFiles) )

    # 对每行进行处理,转化为libffm格式的字符串

    # '标签文件
    fl = os.path.dirname(ffmo) + "/label_" + os.path.basename(ffmo)
    # fl_chk = os.path.dirname(ffmo) + "/chk_" + os.path.basename(ffmo)

    df = pd.read_csv(fi,dtype=data_types,chunksize=2000)

    for k,dc in enumerate(df):
        print(".......chunk" + str(k) + " of " + os.path.basename(fi) + " dealing..." )
        chunkffm=[]
        chunkfm=[]
        # lines =[]
        for i in range(len(dc)):
            line = dc.iloc[i,:]
            ffm,fm=encoder2ffm(line, features_train_dict, feature2field)
            chunkffm.append(ffm)
            chunkfm.append(fm)
            # lines.append(line)

        pd.DataFrame(chunkffm).to_csv(ffmo,mode='a',index=False,header=False)
        pd.DataFrame(chunkfm).to_csv(fmo,mode='a',index=False,header=False)

        ids = dc['id']
        ids.to_csv(fl,mode='a',index=False,header=False)

        # pd.DataFrame(lines).to_csv(fl_chk,mode='a',index=False,header=False)
   
    print(os.path.basename(ffmo) + "....." + " of " + os.path.basename(fi)+ "-------OK!"+ ' Left Files:' + str(TotalFiles) )

if __name__ == '__main__': 

    start0 = time.time()

    j =3
    out_path_temp = "/media/leikun/programs/AI100/AI100-Final/Input/train/" + featrue_version+ "_" + "data/temp" + str(j)
    #修改当前工作目录
    os.chdir(out_path_temp)
    #将该文件夹下的所有文件名存入一个列表
    file_list = os.listdir()

    ffmo = fos1[j]
    fmo = fos2[j]

    if os.path.exists(ffmo):
        os.remove(ffmo)

    if os.path.exists(fmo):
        os.remove(fmo)

    # 标签文件
    fl = os.path.dirname(ffmo) + "/label_" + os.path.basename(ffmo)
    if os.path.exists(fl):
        os.remove(fl)

    TotalFiles = len(file_list)

    param_list =[]
    for f in file_list:
        temp =([out_path_temp +'/'+f,ffmo,fmo],None)
        param_list.append(temp)

    pool = threadpool.ThreadPool(6) # 同时开线程数
    requests = threadpool.makeRequests(make_model_data, param_list) 
    [pool.putRequest(req) for req in requests]   
    pool.wait() # 等待所有线程结束 
    

    end0 = time.time()
    print("Total time use: %d sec." %(end0-start0))
