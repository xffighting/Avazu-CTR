import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures

def poly(train_file, output_file, degree = 2):
    X = pd.read_csv(train_file)
    X_train = X.drop(['id','click'],axis=1)
    poly = PolynomialFeatures(degree=degree)

    X_train = poly.fit_transform(X_train)
    X_train = pd.DataFrame(X_train)
    print(X_train.shape)
    print(X_train.head())
    X_train.to_csv(output_file, index=False)
    return X_train

train_file = '/Users/feixi/Documents/Study/CSDN/Projects/CTR/data/train_50w_rate.csv'
output_file = '/Users/feixi/Documents/Study/CSDN/Projects/CTR/data/train_50w_poly.csv'

poly(train_file=train_file,output_file=output_file)