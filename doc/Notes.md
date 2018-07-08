# 学习记录

## 1. 模型的选择

CTR应该是一个分类模型，并且输出概率。

所以优先考虑的模型应该是LR， Tree-based, DNN

### LR 模型

优点：

> 简单，训练快
>
> 可解释性高，适用于业务场景。 相比DNN，一旦出了问题，无法追溯解释

可以考虑Degree 2 Polynomial mapping

### VW

> 类似LR的线性模型，有很多非线性的变换
>
> Yahoo出品

### RF

> random forest

#### libMF

#### libFM

#### SVDFeature

> 矩阵分解

### 处理CTR 工具的库 

#### liblinear

Pandas 会非常方便，处理小数据级是首选，但是当数据集上G，对计算机内存消耗惊人

所以工业界一般都是用Liblinear来处理数据，把稀疏型数据转为LibSVM类型的数据。

#### 目前用下采样

#### Spark

- MLlib
- ML pipiline 
  - Dataframe
  - Transformer
  - Estimator
  - Pipeline
  - Parameter

## 2. 分析数据

选择训练数据时尽量保证训练数据的均衡

1. 分组统计
> g1.canvas.set_target(‘ipynb'’)
> data.groupby(‘device_type’,{‘CTR’:g1.aggregate.MEAN(‘click'’)})

2. 统计频次
> data[‘C15’].sketch_summury().frequent_items()
>
> 出现次数非常低的，进行onehotencoding会增加非常大的维度
>
> 有时候会单独把只出现一两次的样本，单独拎出来研究，看看是否可以指定特定的rule
>
> 或者把low frequency的类别进行合并为一类
>
> 因为出现的频次太低，学不到东西



## 3. 模型融合

几个模型预测值进行加权求和

权重是通过训练集进行四个分类器的训练。

也就是把新的四个模型的结果作为新的特征，进行训练。

## 4. 特征工程

特征与特征进行&&操作：

如**中国**与**新年**， **美国**与**感恩节**，对用户的点击有着相关性，因为在节日期间，就会有大量的浏览，购买行为。

一般用来两两组合，一般已经够用了