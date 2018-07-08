import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import Imputer

def predict(X_train, y_train, test):
    lr = LogisticRegression(C = 0.01, penalty='l1', verbose=10000,n_jobs=-1)
    lr.fit(X_train, y_train)
    # y_predict = lr.predict(test)
    # y_predict = pd.DataFrame(y_predict)
    # y_predict.to_csv('data/predict.csv',index=False)


    y_predict_prob = lr.predict_proba(test)
    y_predict_prob = pd.DataFrame(y_predict_prob)
    y_predict_prob.to_csv('data/predict_prob.csv',index=False)
    # return y_predict

train_file = pd.read_csv('data/train_50w_rate.csv')
y_train = train_file['click']
X_train = train_file.drop(['click','id'], axis=1)

# test = pd.read_csv('data/test_rate.csv')
test = pd.read_csv('data/test_rate_no_na.csv')
# test.drop(['id'],axis = 1, inplace = True)
# # print(np.isnan(test).any())
# imp = Imputer(strategy='mean')
# test = imp.fit_transform(test)
# test = pd.DataFrame(test)
# test.to_csv('data/test_rate_no_na.csv',index = False)
# print(np.isnan(test).any())
print('start predict:')
predict(X_train=X_train, y_train=y_train, test = test)
