import pandas as pd
import numpy as np
train_df=pd.read_csv('/Users/feixi/Documents/Study/CSDN/Projects/CTR/data/train_200w.csv')
test_df=pd.read_csv('/Users/feixi/Documents/Study/CSDN/Projects/CTR/data/train_1.csv')
#down sampling
temp_0=train_df.click==0
data_0=train_df[temp_0]
temp_1=train_df.click==1
data_1=train_df[temp_1]
data_0_ed=data_0[0:len(data_1)]
data_downsampled=pd.concat([data_1,data_0_ed])
#打乱数据
data_downsampled=data_downsampled.sample(frac=1)
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier()
y = data_downsampled.click #构造y，y为click列
a = list(range(data_downsampled.shape[1]))
del a[1]
X=data_downsampled[a] #构造X，X为除了click列以外的其他列


X = X[[1,2,3,6,9,13,14,15,16,17,18,19,20,21,22]]
X = pd.get_dummies(X)
X[[0,1,2,3,4,5,6,7,8,9,10,11,12]].describe()

gbdt.fit(X, y)

test_labels=test_df.click
test_df=test_df[a]
test_data = test_df[[1,2,3,6,9,13,14,15,16,17,18,19,20,21,22]]
test_data = pd.get_dummies(test_data)
new_test_data = test_data.ix[:,X.columns].fillna(0)
	

# gbdt_predict_labels = gbdt.predict(new_test_data)
gbdt_prob = gbdt_prob(new_test_data)
sum(gbdt_predict_labels==test_labels)
print('First GBDT')
print_metrics(test_labels,gbdt_prob)


new_train_data = X.ix[:,gbdt.feature_importances_>0]
new_test_data = new_test_data.ix[:,gbdt.feature_importances_>0]


n_estimator = 10
from sklearn.model_selection import train_test_split
X_train, X_train_lr, y_train, y_train_lr = train_test_split(new_train_data, y, test_size=0.5)
grd = GradientBoostingClassifier(n_estimators=n_estimator)
#调用one-hot编码
from sklearn.preprocessing import OneHotEncoder
grd_enc = OneHotEncoder()
#调用LR分类模型
from sklearn.linear_model import LogisticRegression
grd_lm = LogisticRegression()
#使用X_train训练GBDT模型，后面用此模型构造特征
grd.fit(X_train, y_train)
#fit one-hot编码器
grd_enc.fit(grd.apply(X_train)[:, :, 0])
#使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(new_test_data)[:, :, 0]))[:, 1]
#预测结果
# y_predict = grd_lm.predict(grd_enc.transform(grd.apply(new_test_data)[:, :, 0]))
print('GBDT + LR')
print_metrics(test_labels,y_pred_grd_lm)


def logloss(act, pred):
    
    '''
    比赛使用logloss作为evaluation
    '''
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def print_metrics(true_values, predicted_values):
    print ("logloss: " + str(logloss(true_values, predicted_values)))
    print ("Accuracy: " + str( metrics.accuracy_score(true_values, predicted_values)))
    print ("AUC: " + str (metrics.roc_auc_score(true_values, predicted_values)))
    print ("Confusion Matrix: "  +  str(metrics.confusion_matrix(true_values, predicted_values)))
    print (metrics.classification_report(true_values, predicted_values))

#使用gridsearchCV选择最优参数
from sklearn.grid_search import GridSearchCV
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
    
    
def find_GBDT_params(x_train, y_train):
    param_GBDT = {
    'max_depth': (3, 5),
    'n_estimators': (50, 100, 200, 300),
    'learning_rate': (0.1, 0.3, 0.5)
    }
    gsearch_GBDT = GridSearchCV(estimator =GradientBoostingClassifier(), param_grid=param_GBDT, cv=10)
    print("Performing grid search...")
    t0 = time()
    gsearch_GBDT.fit(x_train,y_train)
    print("done in %0.3fs" % (time() - t0))
    print()
    #gsearch_GBDT.grid_scores_
    # 输出best score
    print("Best score: %0.3f" % gsearch_GBDT.best_score_)
    print("Best parameters set:")
    print(gsearch_GBDT.best_params_)