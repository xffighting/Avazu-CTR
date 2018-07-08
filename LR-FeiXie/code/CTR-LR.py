import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV

#保留的特征：目前去掉了Hour， 因为下面对hour重新进行的了特征提取
feature_list = ['C1', 'banner_pos', 'site_id', 'site_domain',
       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
       'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
#去掉原来会导致merge值减少的feature ： site_id, site_domain, app_id,
# device_id, device_ip, device_model,device_conn_type,c14
# feature_list = ['C1', 'banner_pos', 
#        'site_category', 'app_domain', 'app_category', 
#        'device_type', 'device_conn_type', 'C14',
#        'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

### 将所有特征的后验概率加入到训练数据集中，并替换掉原来的特征
def insert_click_rate(train_file, rate_file_path, feature_list, output):
    train_rate = pd.read_csv(train_file) 
    print('before merge shape：',train_rate.shape)
    for column in feature_list:
        rate = pd.read_csv(rate_file_path+'clickVS'+column+'.csv', usecols=[column,'avg(click)'])
        train_rate = pd.merge(train_rate,rate,how='left')
        print('after {} merge ,shape:{}'.format(column,train_rate.shape))
        train_rate.rename(columns={'avg(click)':column+'_rate'}, inplace = True)
        train_rate.drop([column],inplace = True, axis=1)
    train_rate.to_csv(output,index=False)
    print('inset rate result shape:', train_rate.shape)
    return train_rate


def ss_feature(df):
    ss_X = StandardScaler()
    ss_df = ss_X.fit_transform(df)

    return ss_df


def mm_feature(df):
    mm_X = MinMaxScaler()
    mm_df = mm_X.fit_transform(df)

    return mm_df

def fit_LR_model(X_train, y_train):
    Cs = [0.01]
    # 大量样本（6W+）、高维度（93），L2正则 --> 缺省用lbfgs
    # LogisticRegressionCV比GridSearchCV快
    lr = LogisticRegressionCV(Cs= Cs, cv = 3, scoring='neg_log_loss', 
                                penalty='l1',solver='liblinear',n_jobs=-1)
    lr.fit(X_train, y_train)
    print("LR scores:",lr.scores_)   

def find_LR_params(x_train, y_train):
    param_LR= {'C':[0.1,1,2]}
    gsearch_LR = GridSearchCV(estimator = LogisticRegression(penalty='l1',solver='liblinear'), param_grid=param_LR,cv=3)
    print("Performing grid search...")
    t0 = time()
    gsearch_LR.fit(x_train,y_train)
    print("done in %0.3fs" % (time() - t0))
    print()
    print("Best score: %0.3f" % gsearch_LR.best_score_)
    print("Best params: ")
    print(gsearch_LR.best_params_)
    #return gsearch_LR.grid_scores_, gsearch_LR.best_params_, gsearch_LR.best_score_
    
from sklearn.externals import joblib

    1

模型保存

>>> os.chdir("workspace/model_save")
>>> from sklearn import svm
>>> X = [[0, 0], [1, 1]]
>>> y = [0, 1]
>>> clf = svm.SVC()
>>> clf.fit(X, y)  
>>> clf.fit(train_X,train_y)
>>> joblib.dump(clf, "train_model.m")


# train_file = 'data/train_1_time_encode.csv' 
#      
# path = 'data/train_info/'
# train_raw = insert_click_rate(train_file = train_file, 
#     rate_file_path = path, feature_list = feature_list, output = 'data/train_1_rate.csv')
# print('train raw shape:',train_raw.shape)

# train_rate_file = pd.read_csv('data/train_1_rate.csv')

# y_train = train_rate_file['click']
# print('y train shape:',y_train.shape)
# X_train = train_rate_file.drop(['click','id'], axis=1)
# print(X_train.head())
# # X_train = ss_feature(X_train)
# X_train = mm_feature(X_train)
# print('train shape:',X_train.shape)
# fit_LR_model(X_train = X_train, y_train = y_train)


## 对50数据集进行验证
# train_file = 'data/train_50w_time_encode.csv' 
# path = 'data/train_info/'
# train_raw = insert_click_rate(train_file = train_file, 
#     rate_file_path = path, feature_list = feature_list, output = 'data/train_50w_rate.csv')
# print('train raw shape:',train_raw.shape)

# train_rate_file = pd.read_csv('data/train_50w_rate.csv')

# y_train = train_rate_file['click']
# print('y train shape:',y_train.shape)
# X_train = train_rate_file.drop(['click','id'], axis=1)
# print(X_train.head())
# print('train shape:',X_train.shape)
# test = pd.read_csv('data/test_rate.csv')
# test.drop(['id'], axis=1, inplace=True)
# predict_y = predict(X_train = X_train, y_train = y_train, test = test)

#生成Test文件的编码文件
# train_file = 'data/test_time_encode.csv' 
# path = 'data/train_info/'
# insert_click_rate(train_file = train_file,rate_file_path = path, feature_list = feature_list, output = 'data/test_rate.csv')

#生成Test文件的编码文件
# train_file = 'data/test_time_encode.csv' 
# path = 'data/train_info/'
# insert_click_rate(train_file = train_file,rate_file_path = path,
# feature_list = feature_list, output = 'data/test_rate.csv')

# X_train = pd.read_csv('/Users/feixi/Documents/Study/CSDN/Projects/CTR/data/train_1_poly.csv')
y_train = pd.read_csv('/Users/feixi/Documents/Study/CSDN/Projects/CTR/data/train_1_rate.csv',usecols=['click'])
# y_train.reshape(-1)
print(y_train.values.reshape(-1))
# print('start predict cv, time:')
# fit_LR_model(X_train=X_train,y_train=y_train)
# print('finish time:')