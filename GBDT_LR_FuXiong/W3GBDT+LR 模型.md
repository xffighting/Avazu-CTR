---
typora-root-url: C:\Users\Administrator\Desktop\img
---

#   			W3:GBDT+LR 模型

作者：付雄（fuxi）

日期：2018/7/7



### W3.1GBDT+LR模型的主要思想

机器学习领域，特征决定模型性能上界。如果能够将数据表达成为线性可分的数据，那么使用简单的线性模型就可以取得很好的效果。而使用GBDT构建新特征的方法，能更好地表达数据的内在联系。参考Facebook[1] ,主要思想是：GBDT每棵树的路径直接作为LR输入特征使用 。例如：

![](/picture-1.png)

图中共有两棵树，x为一条输入样本，遍历两棵树后，x样本分别落到两颗树的叶子节点上，每个叶子节点对应LR一维特征，那么通过遍历树，就得到了该样本对应的所有LR特征。构造的新特征向量是取值0/1的。举例来说：上图有两棵树，左树有三个叶子节点，右树有两个叶子节点，最终的特征即为五维的向量。对于输入x，假设他落在左树第一个节点，编码[1,0,0]，落在右树第二个节点则编码[0,1]，所以整体的编码为[1,0,0,0,1]，这类编码作为特征，输入到LR中进行分类。

### W3.2GBDT+LR模型特征工程

•哈希编码：将string特征  catNum=['site_id','site_domain','site_category','app_id','app_domain','app_category',

'device_id','device_ip','device_model']hash化，具体有strConv()函数实现将字符转换为数字

def strConv(strSrc):
    numDes=[]
    tempStr=[]
    tempStr=strSrc
    for line in range(len(tempStr)):
        numDes.append(hashstr_1(str(tempStr[line]))) #调用hash函数
    return numDes

•日期处理：hour=hour_int + day_week + hour_day

将日期分解为hour_int(开始到现在总的小时数)， day_week(周几)，hour_day(一天的第几个小时)

•增加feature：

  #增加是否是周末特征：is_weekend

fea0_1['is_weekend']=0
fea1_1['is_weekend']=0
fea0_1.ix[fea0_1.day_week.values==6,"is_weekend"]=1
fea1_1.ix[fea1_1.day_week.values==6,"is_weekend"]=1

user_id=device_ip+device_id  #增加user_id特征

device_id_count #增加device_id的次数特征

device_ip_count#增加devic_ip的次数特征

### W3.3单GBTD模型

先使用单一GBTD模型确定logloss的baseline

#### 3.3.1模型调参

1.一轮寻找弱分类器个数

直接调用xgboost内嵌的cv寻找最佳的参数n_estimators，弱分类器的个数n_estimators=157

```
#params = {"objective": "multi:softprob", "eval_metric":"mlogloss", "num_class": 9}
xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,  #数值大没关系，cv会自动返回合适的n_estimators  
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample = 0.5,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        objective= 'binary:logistic',
        seed=3)
modelfit(xgb1, X_train, y_train)
```

![](/output_28_1.png)

2.调整树的参数：max_depth & min_child_weight

(粗调，参数的步长为2；下一步是在粗调最佳参数周围，将步长降为1，进行精细调整)

```
#max_depth 建议3-10， min_child_weight=1／sqrt(ratio_rare_event)
max_depth = range(3,10,2)
min_child_weight = range(1,6,2)
param_test2_1 = dict(max_depth=max_depth, min_child_weight=min_child_weight)
gsearch2_1.fit(X_train , y_train)
```

```
Best: -0.412716 using {'max_depth': 5, 'min_child_weight': 5}
```

![](/output_35_2.png)

3.再次调整弱分类器数目

```
#调整max_depth和min_child_weight之后再次调整n_estimators(6,4)
xgb2_3 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000, 
        max_depth=5,
        min_child_weight=5,
        gamma=0,
        subsample=0.3,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        objective= 'binary:logistic',
        seed=3)
```

![](/output_44_1.png)

4.调整subsample 和 colsample_bytree

   (粗调，参数的步长为0.1；下一步是在粗调最佳参数周围，将步长降为0.05，进行精细调整)

![](/output_53_2.png)

5.调整正则化参数reg_alpha 和reg_lambda

```
#reg_alpha = [1e-3, 1e-2, 0.05, 0.1]    #default = 0
#reg_lambda = [1e-3, 1e-2, 0.05, 0.1]   #default = 1
reg_alpha = [ 1,1.5, 2,2.5,3]    #default = 0, 测试0.1,1，1.5，2
reg_lambda = [0.05,0.1,0.5, 1.5,2]      #default = 1，测试0.1， 0.5， 1，2
param_test5_1 = dict(reg_alpha=reg_alpha, reg_lambda=reg_lambda)
```

![](/output_58_2.png)

```
#最终单GBDT模型参数为：
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 0.7,
 'colsample_bytree': 0.9,
 'gamma': 0,
 'learning_rate': 0.05,
 'max_delta_step': 0,
 'max_depth': 5,
 'min_child_weight': 5,
 'missing': None,
 'n_estimators': 529,
 'nthread': 1,
 'objective': 'binary:logistic',
 'reg_alpha': 2.5,
 'reg_lambda': 0.5,
 'scale_pos_weight': 1,
 'seed': 3,
 'silent': 1,
 'subsample': 0.85}
```

#### 3.3.2kaggle结果

提交结果到kaggle上得到Leaderboard score为：

![](/submission-GBDT.PNG)

### W3.4 GRTD+LR模型训练

#### 3.4.1数据采集

将train.csv随机抽取10w数据。

#### 3.4.2参数设置

#xgboost参数设置

xgb1 = XGBClassifier(
        learning_rate =0.4,
        n_estimators=35,    
        max_depth=8,
        min_child_weight=50, # 叶子节点所需要的最小样本权重和,大小容易过拟合
        gamma=0,
        subsample = 1.0,#0.85
        colsample_bytree=0.5,
        colsample_bylevel=0.9,
        reg_alpha = 2.5,
        reg_lambda= 0.5,
        base_score=0.16,
        objective= 'binary:logistic',
        seed=999)

#lr参数设置

{'C': 0.01, 'penalty': 'l2{'C': 0.01, 'penalty': 'l2'}

0.4058299286400423 #交叉验证logloss

提交结果到kaggle上Leaderboard score：

![](/submission-GBDT_LR.PNG)

#### 3.4.3结果分析

GBDT+LR模型的结果，并没有详细调参的单模型的结果好，模型融合后，性能反而下降了，模型处于欠拟合状态，与kaggle上排名靠前的logloss相比还有很大的距离，具体原因可能为：1.数据的分布是不均衡的，需要考虑，但是自己想到这里时，已没有时间从头做，而且数据的采样不全，并没有包括所有特质的取值情况；2.没有从特征选择的角度出发，从而构建出更多的特征。

#### 3.4.4总结

1. 提交结果时，test.csv的id列为float类型，而submission的id列类型为int型，在kaggle上提交结果总是提示错误，找了好久才发现这个错误，浪费了好多时间；
2. 训练集，验证集和测试集的各个特征取值不完全一样，在进行编码的时候要统一考虑（由于test文件较大，内存会溢出，8G内存运气好会编码通过，可以重启kernel，重新运行）；
3. 需要理解增加特征的物理意义，如果能转化数学表达式，则能进行数学推导，理论上证明可行性，否则只能凭感觉，靠运气；
4. 如何先构建最简单的模型，跑出结果，从结果出发分析，那些特征对结果影响较大，什么特征导致正确率的下降较快，还应该从var和bias的角度分析模型的性能，这些将是自己以后所需要学习。

### W3.5参考文献

[1]. Xinran He et al. Practical Lessons from Predicting Clicks on Ads at Facebook, 2014.

[2].  http://www.cbdio.com/BigData/2015-08/27/content_3750170.htm
[3].  https://blog.csdn.net/shine19930820/article/details/71713680
[4].  https://www.zhihu.com/question/35821566
[5].  https://github.com/neal668/LightGBM-GBDT-LR/blob/master/GBFT%2BLR_simple.py

 

 

 

 

 



