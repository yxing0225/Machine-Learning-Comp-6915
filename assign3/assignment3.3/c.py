# -*- coding: UTF-8 -*-
import sys
import csv
import numpy as np
import pandas as pd
np.set_printoptions(threshold=1e6)
from numpy import *
from sklearn import datasets
#from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from operator import itemgetter
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

"""
Description:新准备数据，可以把数据变为特征feature和分类targets两个数组

Parameters:
    filename - the name of the file 
Returns:
    tf - return the content of file as a list,and use tab to split
"""
def prepare_data(train_data):
    train_data = array(train_data)
    features = train_data[:, :-1]
    targets = train_data[:, -1:]
    print(targets)

    return features, targets

def prepare_testdata(test_data):
    train_data = array(test_data)
    features = train_data[:, :]
    testfeatures = []
    transfer = []
    for i in features[1:]:
        for j in i:
            j = float(j)
            transfer.append(j)
        testfeatures.append(transfer)
        transfer = []
    return testfeatures

"""
    函数说明:用来切分数据集

    Parameters:
        tf_test_each - 测试集,
        tf_train_all - 训练集
        distance -一个测试元素与一个训练元素的距离
        distancesetall  - 所有测试元素与所有训练元素的距离集合
        namedistanceset - 名字+距离的list
    Returns:
        返回所有测试元素与所有训练元素的距离集合
    Modify:
        2017-03-24 trainset以前为XY共同的集合，现在分开为featrue和target两个集合
"""
def splitDataSet(feature,target,split_size):
    each_size = (len(feature)-1)/split_size  #一共k折，所以没折就有总数/k多个元素
    #print(each_size)
    feature_each_split = []  #每次添加一个元素进去，添加满一个后。几个组合形成全部分隔好的训练集
    feature_split_all = []
    target_each_split = []  # 每次添加一个元素进去，添加满一个后。几个组合形成全部分隔好的训练集
    target_split_all = []
    count_num = 0   #统计每次遍历当前的个数  第一次计算feature的组合
    count_split = 0   #统计切分次数
    transfer = []  # 为了存储把里面的数据转化为float类型
    for i in feature[1:]:  #遍历所有的折feature的类型为：[array(['overlaps_sQTLs', 'overlaps_sQTLs_4_same_gene', 'gene_has_sQTLs','TTC', 'TTG', 'TTT'], dtype='<U26'),
                                                   # [['0.0', '0.0', '1.0', '0.051948052', '0.025974026', '0.077922078'],...所以要从1开始，0为名字
        for j in i:   #遍历没折里面的所有元素
            j = float(j)
            transfer.append(j)
        count_num = count_num + 1
        feature_each_split.append(transfer)   #里面是k折分后，其中一折的的数据float类型
        transfer = []
        if count_num >= each_size:   #如果一折里面的元素个数达到了上线
            feature_split_all.append(feature_each_split)   #把这个折加到总的feature组合里面
            feature_each_split = []
            count_num = 0

    count_num = 0  # 统计每次遍历当前的个数   第二次计算target的组合  从这里是target集合
    count_split = 0  # 统计切分次数
    transfer = []  #transfer the word to float
    for i in target[1:]:
        for j in i:
            j = float(j)
            transfer.append(j)
        target_each_split.append(transfer)
        count_num = count_num + 1
        transfer = []
        if count_num >= each_size:
            target_split_all.append(target_each_split)
            target_each_split = []
            count_num = 0

    feature_split_all.insert(0,feature[0])  #把名字加到feature集合里面.组合后的类型为：[array(['overlaps_sQTLs', 'overlaps_sQTLs_4_same_gene', 'gene_has_sQTLs','TTC', 'TTG', 'TTT'], dtype='<U26'),
                                                                        #                [[0.0, 0.0, 1.0, 0.051948052, 0.025974026, 0.077922078],
                                                                        #                 [1.0, 1.0, 1.0, 0.026239067, 0.025614327, 0.05997501],
                                                                        #                 [0.0, 0.0, 0.0, 0.014770459, 0.025149701, 0.033133733],
                                                                        #                 [0.0, 0.0, 0.0, 0.018461538, 0.02, 0.048307692]]
                                                                        #                [[0.0, 0.0, 0.0, 0.022330097, 0.027669903, 0.055339806],
                                                                        #                 [0.0, 0.0, 1.0, 0.033950617, 0.030864198, 0.070987654], ....
    #print("feature_split_all")
    #print(feature_split_all)
    target_split_all.insert(0,target[0])    #把名字加到target集合里面
    #print("target_split_all")
    #print(target_split_all)
    return feature_split_all,target_split_all

"""
    函数说明:用来生成测试集和训练集，把所有的元素全部切成k折后，交叉验证需要把他们在组合起来。选取其中的一部分作为测试集，剩下的组合成为验证集。
            把j个当做为测试机，剩下的组合变为训练结合。并且分为feature和targets两个部分，也就是最终组成四个
            集合feature_testset, feature_trainset, target_testset2, target_trainset2

    Parameters:
        tf_test_each - 测试集,
        tf_train_all - 训练集
        distance -一个测试元素与一个训练元素的距离
        distancesetall  - 所有测试元素与所有训练元素的距离集合
        namedistanceset - 名字+距离的list
    Returns:
        返回所有测试元素与所有训练元素的距离集合
    Modify:
        2017-03-24
        把splitDate转化为两个变量feature和target
"""
def denerageDataSet(feature,target,j):   #第j个数是测试集，其余为训练集.j要从1开始
    feature_testset = []
    target_testset = []
    feature_trainset =[]
    target_trainset = []
    i = 0
    for i in feature[1:]:  #遍历分折后feature里面的所有元素集合一共k个（因为是k折）
        if i is feature[j]:  #如果是第j个元素集合，j当做测试集合赋值给feature_testset。                             [[0.0, 0.0, 1.0, 0.051948052, 0.025974026, 0.077922078],
                                                                                                    #          [1.0, 1.0, 1.0, 0.026239067, 0.025614327, 0.05997501],
                                                                                                    #          [0.0, 0.0, 0.0, 0.014770459, 0.025149701, 0.033133733],
                                                                                                    #          [0.0, 0.0, 0.0, 0.018461538, 0.02, 0.048307692]]
            feature_testset = i
        else:
            feature_trainset = feature_trainset + i  #剩下的所有集合加在一起成为一个训练集合feature_trainset.类型为：[[0.0, 0.0, 0.0, 0.022330097, 0.027669903, 0.055339806],
                                                                                       #                       [0.0, 0.0, 1.0, 0.033950617, 0.030864198, 0.070987654],
                                                                                       #                       [0.0, 0.0, 1.0, 0.021455939, 0.019923372, 0.050574713],
                                                                               #                               [0.0, 0.0, 1.0, 0.021240442, 0.020390824, 0.048428207],
                                                                               #                               [0.0, 0.0, 1.0, 0.020216963, 0.029092702, 0.065581854],
                                                                               #                               [0.0, 0.0, 1.0, 0.019992686, 0.020846032, 0.048518835],
                                                                               #                               [0.0, 0.0, 1.0, 0.016666667, 0.044444444, 0.122222222],
    i = 0
    for i in target[1:]:
        if i is target[j]:
            target_testset = i
        else:
            target_trainset = target_trainset + i
    #print("feature_testset:",feature_testset)
    #print("feature_trainset:",feature_trainset)
    target_testset2 = []
    for i in target_testset:     #为了破开list。之前为[[0.0], [1.0], [0.0], [1.0]], [[0.0], [1.0], [1.0], [1.0]], [[1.0]]。
                                  #破开后，只有一个list,方便处理。[0.0, 1.0, 0.0, 1.0]
        a = i[0]
        target_testset2.append(a)
    target_trainset2 = []
    for i in target_trainset:
        a = i[0]
        target_trainset2.append(a)
    #print("target_testset2:", target_testset2)
    #print("target_trainset2:", target_trainset2)
    return feature_testset,feature_trainset,target_testset2,target_trainset2

"""
    Description:figure out the distance of testset and trainset

    Parameters:
        tf_test_each - the testset
        tf_train_all - the train set
        ##distance - the distance between a testset factor and a trainfact factor
        ##istanceset - list like [[distance1,name1],[distance2,name2],...]
    Returns:
        distancesetall  - the set of distance between all test factors and all train factors[[[distance1,name1],[distance2,name2],...][[distance1,name1],[distance2,name2],...[distancen,namen]]...]
    Modify:
        2017-03-24
"""
def KNNcompare(feature_testset,feature_trainset,target_trainset,target_testset,kn): # figure out the distance of testset and trainset
    # 定义分类器，并且选用邻居数为kn
    knn = KNeighborsClassifier(n_neighbors = kn)
    # 进行分类，自带功能。直接可以训练。输入Xtrain和Ytrain
    knn.fit(feature_trainset, target_trainset)
    # 计算预测值.输入Xtest就可以直接预测出结果
    y_predict = knn.predict(feature_testset)
    #print("the predict is: ",y_predict)
    #print(target_testset)
    predict_proba = knn.predict_proba(feature_testset)  #可以打印出预测1的概率和0的概率
    #print("the probability: ",predict_proba)
    distance = 0
    i = 0
    #sum = 0
    while i < len(y_predict):    #遍历要预测的数值的个数遍。
        if y_predict[i]:   #如果预测数值和真实值不等，则差距加1
            distance = distance + 1
        i = i + 1
    #print(distance)       #打印出当前k个邻居的情况下，在交叉严重这一种验证集的情况下的距离
    return distance,predict_proba

def KNN(splitfeatures,splittargets,kn):
    distancesum = 0
    distance = []
    distancesum = 0
    a = 1   #a要从1开始，一共5折，循环5次
    for i in kn:   #kn为邻居个数的集合，为了选出最佳的k值
        #print("测试的K是",i)
        while a <= 5:       #使用5折验证，在当前k个邻居的情况下，求出5次的距离。求平均值，就可以求出当前k个邻居下，预测数值和这是指的差距平均值
            #print("第",a,"折计算")
            feature_testset,feature_trainset,target_testset,target_trainset = denerageDataSet(splitfeatures,splittargets,a)
            distancesum = distancesum + KNNcompare(feature_testset, feature_trainset, target_trainset,target_testset,i)[0]  #把求出来的距离求和，求平均值
            a = a + 1
        distancemean = distancesum/5
        distance.append(distancemean)  #把差距加到差距list里面，为了找到list里面的最下差距，确定最好的邻居数
        distancesum = 0
        distancemean = 0
        a = 1

    #print("distance:",distance)  #差距list为:(第1个k对应的差距，第2个k对应的差距，第3个k对应的差距)
    j = 0
    bestkn = kn[j]
    while j < (len(distance)-1):
        if distance[j+1] < distance[j]:   #比较那个差距list里面那个差距是最小的，并返回这个差距对应的邻居个数
            bestkn = kn[j+1]
        j = j + 1
    #print("bestkn",bestkn)
    return bestkn         #返回最好的邻居个数，使用交叉验证确定了超参k

def KNNfinal(allfeature_trainset,alltarget_trainset,allfeature_testset,kn):  #带入所有训练featreu和target，和测试feature,以及最佳邻居个数
    knn = KNeighborsClassifier(n_neighbors=kn)
    # 进行分类
    knn.fit(allfeature_trainset, alltarget_trainset)
    # 计算预测值
    y_predict = knn.predict(allfeature_testset)
    #print("the predict is: ", y_predict)
    predict_proba = knn.predict_proba(allfeature_testset)   #得到结果概率矩阵
    #print(predict_proba)

    predict_proba_1 = []
    for i in predict_proba:    #只保留结果为1类的概率
        predict_proba_1.append(i[1])
    #predict_proba_1.insert(0,"结果为1的概率")
    #for i in predict_proba_1:   #变为纵向输出
    #    print(i)    #打印出接近1的概率
    return y_predict, predict_proba_1


def logicrcompare(feature_trainset,target_trainset,feature_testset,target_testset,n):   #feature_trainset是训练feature的集合，n表示要读取多少个feature,
    feature_trainset = np.array(feature_trainset)
    x = feature_trainset[:, 0:n]
    #print(x)

    logicclassifier = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression())])  #搭建模型，训练LogisticRegression分类器
    # 开始训练
    #print("target_trainset.ravel",target_trainset)
    #print("feature_trainset",feature_trainset)
    #print("target_trainset",target_trainset)
    logicclassifier.fit(feature_trainset, target_trainset)
    #print("训练完毕")
    y_predict = logicclassifier.predict(feature_testset)
    #print("the predict is: ",y_predict)
    #print(target_testset)
    predict_proba = logicclassifier.predict_proba(feature_testset)  # 可以打印出预测1的概率和0的概率
    # print("the probability: ",predict_proba)
    distance = 0
    i = 0
    # sum = 0
    while i < len(y_predict):  # 遍历要预测的数值的个数遍。
        if y_predict[i] != target_testset[i]:  # 如果预测数值和真实值不等，则差距加1
            distance = distance + 1
        i = i + 1
    #print(distance)       #打印出当前k个邻居的情况下，在交叉严重这一种验证集的情况下的距离
    return distance, predict_proba

def logicr(splitfeatures,splittargets,featurenum):
    distancesum = 0
    distance = []
    distancesum = 0
    a = 1   #a要从1开始，一共5折，循环5次
    for i in featurenum:   #featurenum为所要选择的特征的个数的集合，为了选出最佳的特征个数
        #print("测试的特征个数是",i)
        while a <= 5:       #使用5折验证，在当前k个邻居的情况下，求出5次的距离。求平均值，就可以求出当前k个邻居下，预测数值和这是指的差距平均值
            #print("第",a,"折计算")
            feature_testset,feature_trainset,target_testset,target_trainset = denerageDataSet(splitfeatures,splittargets,a)
            distancesum = distancesum + logicrcompare(feature_trainset, target_trainset,feature_testset, target_testset,i)[0]  #把求出来的距离求和，求平均值
            a = a + 1
        distancemean = distancesum/5
        distance.append(distancemean)  #把差距加到差距list里面，为了找到list里面的最下差距，确定最好的邻居数
        distancesum = 0
        distancemean = 0
        a = 1

    #print("distance:",distance)  #差距list为:(第1个k对应的差距，第2个k对应的差距，第3个k对应的差距)
    j = 0
    bestfeaturenum = featurenum[j]
    while j < (len(distance)-1):
        if distance[j+1] < distance[j]:   #比较那个差距list里面那个差距是最小的，并返回这个差距对应的邻居个数
            bestfeaturenum = kn[j+1]
        j = j + 1
    #print("bestfurenum",bestfeaturenum)
    return bestfeaturenum         #返回最好的邻居个数，使用交叉验证确定了超参k

def logicrfinal(allfeature_trainset,alltarget_trainset,allfeature_testset,featurenum):  #带入所有训练featreu和target，和测试feature,以及最佳feature个数
    logicclassifier = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression())])
    # 进行分类
    logicclassifier.fit(allfeature_trainset, alltarget_trainset)
    # 计算预测值
    y_predict = logicclassifier.predict(allfeature_testset)
    #print("the predict is: ", y_predict)
    predict_proba = logicclassifier.predict_proba(allfeature_testset)   #得到结果概率矩阵
    #print(predict_proba)
    #return y_predict
    predict_proba_1 = []
    for i in predict_proba:    #只保留结果为1类的概率
        predict_proba_1.append(i[1])
    #predict_proba_1.insert(0,"结果为1的概率")
    #for i in predict_proba_1:   #变为纵向输出
    #    print(i)    #打印出接近1的概率
    return y_predict, predict_proba_1

def treecompare(feature_testset,feature_trainset,target_trainset,target_testset,dep): # 树的最大深度为dep
    feature_trainset = np.array(feature_trainset)
    # 定义分类器，并且选取树的深度为depth
    clf = tree.DecisionTreeClassifier(max_depth=dep)
    # 进行分类，自带功能。直接可以训练。输入Xtrain和Ytrain
    print(feature_trainset)
    print(target_trainset)
    clf.fit(feature_trainset, target_trainset)
    # 计算预测值.输入Xtest就可以直接预测出结果

    y_predict = clf.predict(feature_testset)
    print("the predict is: ",y_predict)
    print(target_testset)
    predict_proba = clf.predict_proba(feature_testset)  #可以打印出预测1的概率和0的概率
    print("the probability: ",predict_proba)
    distance = 0
    i = 0
    #sum = 0
    while i < len(y_predict):    #遍历要预测的数值的个数遍。
        if y_predict[i] != target_testset[i]:   #如果预测数值和真实值不等，则差距加1
            distance = distance + 1
        i = i + 1
    print(distance)       #打印出当前dep深度的情况下，在交叉严重这一种验证集的情况下的距离
    return distance,predict_proba

def dtree(splitfeatures,splittargets,depth):
    distancesum = 0
    distance = []
    distancesum = 0
    a = 1   #a要从1开始，一共5折，循环5次
    for i in depth:   #depth为所要选择的特征的个数的集合，为了选出最佳的树的深度
        #print("测试的特征个数是",i)
        while a <= 5:       #使用5折验证，在当前k个邻居的情况下，求出5次的距离。求平均值，就可以求出当前k个邻居下，预测数值和这是指的差距平均值
            #print("第",a,"折计算")
            feature_testset,feature_trainset,target_testset,target_trainset = denerageDataSet(splitfeatures,splittargets,a)
            distancesum = distancesum + logicrcompare(feature_trainset, target_trainset,feature_testset, target_testset,i)[0]  #把求出来的距离求和，求平均值
            a = a + 1
        distancemean = distancesum/5
        distance.append(distancemean)  #把差距加到差距list里面，为了找到list里面的最下差距，确定最好的邻居数
        distancesum = 0
        distancemean = 0
        a = 1

    #print("distance:",distance)  #差距list为:(第1个dep对应的差距，第2个dep对应的差距，第3个dep对应的差距)
    j = 0
    bestdep = depth[j]
    while j < (len(distance)-1):
        if distance[j+1] < distance[j]:   #比较那个差距list里面那个差距是最小的，并返回这个差距对应的树的深度
            bestdep = kn[j+1]
        j = j + 1
    #print("bestfurenum",bestfeaturenum)
    return bestdep         #返回最好的邻居个数，使用交叉验证确定了超参k

def dtreefinal(allfeature_trainset,alltarget_trainset,allfeature_testset,dep):  #带入所有训练featreu和target，和测试feature,以及最佳feature个数
    clf = tree.DecisionTreeClassifier(max_depth=dep)
    # 进行分类
    clf.fit(allfeature_trainset, alltarget_trainset)
    # 计算预测值
    y_predict = clf.predict(allfeature_testset)
    #print("the predict is: ", y_predict)
    predict_proba = clf.predict_proba(allfeature_testset)   #得到结果概率矩阵
    #print(predict_proba)

    predict_proba_1 = []
    for i in predict_proba:    #只保留结果为1类的概率
        predict_proba_1.append(i[1])
    #predict_proba_1.insert(0,"结果为1的概率")
    #for i in predict_proba_1:   #变为纵向输出
    #    print(i)    #打印出接近1的概率

    return y_predict,predict_proba_1

'''
没用的函数
'''


def bestmodel(splitfeatures,splittargets):
    feature_testset, feature_trainset, target_testset, target_trainset = denerageDataSet(splitfeatures, splittargets,1)  # 多余步骤，生成交叉验证需要的四个集合，然后两两结合起来，结合为全部的trainfeature和traintarget训练集合
    allfeature_trainset = feature_testset + feature_trainset  # 全部的trainfeatures
    alltargets_trainset = target_testset + target_trainset  # 全部的traintargets
    kn = [1, 3, 5]  # 给定KNN可以选择的邻居个数
    k = KNN(splitfeatures, splittargets, kn)  # 选定一个最好的邻居个数，使用交叉验证
    q = KNNfinal(feature_trainset, target_trainset, feature_testset,k)[0]  # 使用KNN分类器，对测试数据进行分类得出。用上一步求出来的超参k，求出来测试训练的概率结果
    #print(q)
    #print(target_testset)

    distance = 0
    KNNdistance = 0
    i = 0
    while i < len(target_testset):  # 遍历要预测的数值的个数遍。
        if q[i] != target_testset[i]:  # 如果预测数值和真实值不等，则差距加1
            distance = distance + 1
        i = i + 1
    KNNdistance = KNNdistance + distance
    #print("KNNdistance:", KNNdistance)
    print("================================================================")
    feature_testset, feature_trainset, target_testset, target_trainset = denerageDataSet(splitfeatures, splittargets, 1)
    featurenum = [2, 3, 4, 5]
    num = logicr(splitfeatures, splittargets, featurenum)
    w = logicrfinal(feature_trainset, target_trainset, feature_testset, num)[0]
    #print(w)
    #print(target_testset)
    distance = 0
    logicrdistance = 0
    i = 0
    while i < len(target_testset):  # 遍历要预测的数值的个数遍。
        if w[i] != target_testset[i]:  # 如果预测数值和真实值不等，则差距加1
            distance = distance + 1
        i = i + 1
    logicrdistance = logicrdistance + distance
    #print("logicrdistance:", logicrdistance)
    print("================================================================")
    feature_testset, feature_trainset, target_testset, target_trainset = denerageDataSet(splitfeatures, splittargets, 1)
    depth = [1, 2, 3, 4]  # 给定树的深度
    dep = dtree(splitfeatures, splittargets, depth)  # 选定一个最好的树的深度，使用交叉验证
    e = dtreefinal(feature_trainset, target_trainset, feature_testset,dep)[0]  # 使用treeclassfier分类器，对测试数据进行分类得出。用上一步求出来的超参dep，求出来测试训练的概率
    #print(e)
    #print(target_testset)
    distance = 0
    dtreedistance = 0
    i = 0
    while i < len(target_testset):  # 遍历要预测的数值的个数遍。
        if e[i] != target_testset[i]:  # 如果预测数值和真实值不等，则差距加1
            distance = distance + 1
        i = i + 1
    dtreedistance = dtreedistance + distance
    #print("dtreedistance:",dtreedistance)
    distancesum = {'KNNdistance':KNNdistance,"logicrdistance":logicrdistance,'dtreedistance': dtreedistance}
    print("===========================")
    #distance(splitfeatures,splittargets,k,num,dep)
    #print(distancesum)
    #distanceorder = sorted(distancesum.items(),key=lambda item:item[1])
    a = []
    for i in distancesum.keys():
        a.append(i)
    #print(a)
    if a[0] == 'KNNdistance':
        return 1 , k
    elif a[0] == 'logicrdistance':
        return 2 , num
    else:
        return 3 , dep





"""
Description:main function

Parameters:
    none
Returns:
    none

"""
if __name__ == '__main__':
    from ipdb import set_trace

    observation_file = "A3_training_dataset.tsv"
    test_file = "A3_test_dataset.tsv"
    observations = []
    test = []
    with open(observation_file) as tsv:  #打开训练文件文件
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            observations.append(row)
    with open(test_file) as tsv:    #这个循环是用来提取测试集合的
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            test.append(row)
    # print(observations)
    #print("==============================================")
    features, targets = prepare_data(observations)    #用来生成交叉验证用的features 和targets集合
    testfeatures = prepare_testdata(test)     #用来生成测试用的feature集合
    #print("testdata_features:", testfeatures)
    #print(features)
    #print(targets)
    #print("==============================================")
    splitfeatures,splittargets = splitDataSet(features, targets, 10)    #5折切分，每个集合里面一共五个list在加一个名字
    feature_testset, feature_trainset, target_testset, target_trainset = denerageDataSet(splitfeatures, splittargets,1)  # 多余步骤，生成交叉验证需要的四个集合，然后两两结合起来，结合为全部的trainfeature和traintarget训练集合
    allfeature_trainset = feature_testset + feature_trainset  # 全部的trainfeatures
    alltargets_trainset = target_testset + target_trainset  # 全部的traintargets
    choice = bestmodel(splitfeatures,splittargets)
    if choice[0] == 1:
        print("KNN is the best:")
        result = KNNfinal(allfeature_trainset, alltargets_trainset, testfeatures, choice[1])[1]
        output = open('model_selection_table.txt', 'w', encoding='gbk')
        output.write('\t\tKNN is the best\n')
        for i in range(len(result)):
                output.write(str(result[i]))  # write函数不能写int类型的参数，所以使用str()转化
                output.write('\t')  # 相当于Tab一下，换一个单元格
        output.close()
        print("the result is in model_selection_table.txt")
    elif choice[0] == 2:
        print("logicr is the best:")
        result = logicrfinal(feature_trainset, target_trainset, feature_testset, choice[1])[1]
        output = open('model_selection_table.txt', 'w', encoding='gbk')
        output.write('\t\tlogicr is the best:\n')
        for i in range(len(result)):
                output.write(str(result[i]))  # write函数不能写int类型的参数，所以使用str()转化
                output.write('\t')  # 相当于Tab一下，换一个单元格
        output.close()
        print("the result is in model_selection_table.txt")
    else:
        print("decision tree is the best:")
        result = dtreefinal(feature_trainset, target_trainset, feature_testset, choice[1])[1]
        output = open('model_selection_table.txt', 'w', encoding='gbk')
        output.write('\t\tdecision tree is the best:\n')
        for i in range(len(result)):
                output.write(str(result[i]))  # write函数不能写int类型的参数，所以使用str()转化
                output.write('\t')  # 相当于Tab一下，换一个单元格
        output.close()
        print("the result is in model_selection_table.txt")


