# -*- coding: UTF-8 -*-
import sys
import csv
import numpy as np
import pandas as pd
np.set_printoptions(threshold=1e6)
from numpy import *
from sklearn.linear_model import LogisticRegression
from operator import itemgetter
from matplotlib import pyplot

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

    return features, targets

def prepare_testdata(test_data):
    train_data = array(test_data)
    features = train_data[:, :-1]
    print("testdata_features:",features)

    return features

"""
Description:open the TF_output file, and store in a list.

Parameters:
    filename - the output file name.
Returns:
    tf - return the TF_output as a 2D list
"""
def outputsequence(filename): # open output file and store the file content as a list.Y_train.txt will use this function to open.
    i = 0

    with open(filename) as f:
        tf = []

        for line in f.readlines():
            tf_sq = line
            tf_sq_sp = tf_sq.split('\t')
            new_tf_sq_sp = [tf_sq_sp.replace('"', '') for tf_sq_sp in tf_sq_sp]
            tf.append(new_tf_sq_sp)
    return tf

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
    each_size = (len(feature)-1)/split_size
    #print(each_size)
    feature_each_split = []  #每次添加一个元素进去，添加满一个后。几个组合形成全部分隔好的训练集
    feature_split_all = []
    target_each_split = []  # 每次添加一个元素进去，添加满一个后。几个组合形成全部分隔好的训练集
    target_split_all = []
    count_num = 0   #统计每次遍历当前的个数  第一次计算feature的组合
    count_split = 0   #统计切分次数
    transfer = []  # transfer the word to float
    for i in feature[1:]:
        for j in i:
            j = float(j)
            transfer.append(j)
        count_num = count_num + 1
        feature_each_split.append(transfer)
        transfer = []
        if count_num >= each_size:
            feature_split_all.append(feature_each_split)
            feature_each_split = []
            count_num = 0

        #print("===========================")
    #feature_split_all.append(feature_each_split)

    count_num = 0  # 统计每次遍历当前的个数   第二次计算target的组合
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

        # print("===========================")
    #target_split_all.append(target_each_split)
    feature_split_all.insert(0,feature[0])
    print("feature_split_all")
    print(feature_split_all)
    target_split_all.insert(0,target[0])
    print("target_split_all")
    print(target_split_all)
    return feature_split_all,target_split_all
"""
    函数说明:用来生成测试集和训练集

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
"""
def denerageDataSet(splitData,j):   #第j个数是测试集，其余为训练集
    testset = []
    trainset =[]
    i = 0
    for i in splitData:
        if i is splitData[j]:
            testset = i
        else:
            trainset = trainset + i
    #print("testset:",testset)
    #print("trainset:",trainset)
    return testset,trainset

"""
    函数说明:用来训练效果（把训练数据分k组后，其中一种情况（1和k-1的情况）的平均d）目标是求所有情况的d的平方相加，求出spearman

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
"""
def performanceeachtrainset(k,doneset,outputset,points):
    c = comparetf(doneset[0],doneset[1],points)   #donset[0]是测试集，doneset[1]是训练集合
    e = classify(int(k),outputset,c,doneset[0])
    #print(e)
    w = 0
    i = 0
    j = 1
    d1 = 0   #曼哈顿距离
    d2 = 0   #欧氏距离
    dsquaresum1 = 0
    dsquaresum2 = 0
    spearman1 = 0
    spearman2 = 0
    spearmansum1 = 0
    spearmansum2 = 0
    allspearmanmean1 = 0
    allspearmanmean2 = 0
    while w < len(e):   #遍历预测结果结合的所有名字
        while i < len(outputset[0]):   #去整体输出Y里面遍历所有
            if e[w][0] == outputset[0][i]:   #如果测试集里面的名字 找到了在Y里面
                while j < len(outputset):
                    #print("第一次")
                    d1 = float(outputset[j][i]) - e[w][j]   #曼哈顿距离
                    d2 = (float(outputset[j][i])*float(outputset[j][i]) - e[w][j]*e[w][j])**0.5 #欧氏距离
                    #print("输出d：")
                    #print(d)
                    dsquaresum1 = dsquaresum1 + (d1 * d1)
                    dsquaresum2 = dsquaresum2 + (d2 * d2)

                    #print("输出dsquaresum：")
                    #print(dsquaresum)
                    j = j + 1
                spearman1 = 1 - 6 * dsquaresum1 / ((len(outputset) -1 ) * (len(outputset) - 1) * (len(outputset) - 1) - (len(outputset) - 1))
                spearman2 = 1 - 6 * dsquaresum2 / ((len(outputset) - 1) * (len(outputset) - 1) * (len(outputset) - 1) - (len(outputset) - 1))
                #print("输出spearman：")
                #print(spearman)
                j = 1
                #print("第二层一次")
            i = i + 1
            #print("第二层一次")
        i = 0
        spearmansum1 = spearmansum1 + spearman1
        spearmansum2 = spearmansum2 + spearman2
        w = w + 1
    #print(len(e))
    allspearmanmean1 = spearmansum1/len(e)
    allspearmanmean2 = spearmansum2 / len(e)
    return allspearmanmean1,allspearmanmean2


"""
    函数说明:

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
"""
def performancealltrainset(k,splitDate,outputset,points):
    allspearmanmean1 = 0
    allspearmanmean2 = 0
    allspearmansum1 = 0
    allspearmansum2 = 0
    spearmanset1 = []
    spearmanset2 = []
    squaresum1 = 0
    squaresum2 = 0
    for j in range(len(splitDate)):
        a = denerageDataSet(splitDate, j)
        spearman = performanceeachtrainset(k,a,outputset,points)  #求出其中分折后其中一折的spearman
        spearmanset1.append(spearman[0]) #把他加入spearman列表中,列表中为所有折的spearman相关.曼哈顿距离
        spearmanset2.append(spearman[1])  # 把他加入spearman列表中,列表中为所有折的spearman相关.欧氏距离
        #print(spearman)
        allspearmansum1 = allspearmansum1 + spearman[0]  #求出k折后，所有spearman相关的和.曼哈顿距离
        allspearmansum2 = allspearmansum2 + spearman[1]  # 求出k折后，所有spearman相关的和.欧氏距离
    allspearmanmean1 = allspearmansum1 / len(splitDate)  #求出k折后，spearman的平均数.曼哈顿距离
    allspearmanmean2 = allspearmansum2 / len(splitDate)  # 求出k折后，spearman的平均数.欧氏距离
    for i in spearmanset1:
        squaresum1 = squaresum1 + (i-allspearmanmean1)*(i-allspearmanmean1)   #求出方差.曼哈顿距离
    sme1 = squaresum1**0.5
    for w in spearmanset2:
        squaresum2 = squaresum2 + (i-allspearmanmean2)*(i-allspearmanmean2)   #求出方差.欧氏距离
    sme2 = squaresum2**0.5
    #print(allspearmanmean,"+-",sme)
    return allspearmanmean1,sme1
    #performanceeachtrainset(doneset, outputset)

"""
    函数说明:自己设定K和distance function，完全展示

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
"""

def circileperformance():
    k = [2,3,5]
    d = [[2,4,5,24,30,43,45,46,47],[2,4,5,24]]
    row = []
    rowall = []
    bestk = 0
    bestd = 0
    best = 0
    final = []
    for i in k:
        row.append("K=")
        row.append(i)
        for j in d:
            #print('第',i,"   ",j,"次")
            #performanceeachtrainset(i, g, outputset,j)
            a = performancealltrainset(i, f, outputset,j)
            if a[0] > best:
                bestk = i
                bestd = d.index(j)
            row.append(round(a[0],3))
            row.append("+-")
            row.append(round(a[1],3))
            row.append("    ")
        rowall.append(row)
        row = []
    #a = "model chosen: "
    #rowall.append(a)
    #rowall.append(bestk)
    #rowall.append(bestd)
    final.append("model chosen:")
    final.append("\tK=")
    final.append(bestk)
    final.append("\tD=")
    final.append(bestd)

    #final.append(bestd)
    #print("model chosen: K= ",bestk,"    distance = ",bestd)
    rowall.append(final)
    #rowall.append(a)
    return rowall








"""
    Description:according to the test set to find the distance between each factor in test set and each factor in train set. according to K,
            we can find the k nearest neighbors. Figure out the average of neighbors' output.

    Parameters:
        k - you will find the k nearest neighbor
        outputset - list transfer from TF_output.txt [[name1,name2,name3,name4...],[num1,num2,num3,num4...],[num1,num2,num3,num4...]...]
        nameset - list of diatance and name eg:   [[distance1,name1],[distance2,name2],[distance3,name3]...]
        tf_test_all - the testset
    Returns:
        averagesetall - all of test factors' output

"""

"""
Find the distance between each factor in the X_unseen set and each factor in the training set according to the X_unseen set. 
Then based on the K entered by the user, we can find the k nearest neighbors. Find the average output of the neighbors.
"""
def classify(k,outputset,nameset,tf_test_all):
    i= 0
    j = 0
    z = 1
    w = 0
    target = []
    sum = 0
    average = 0
    averageset = []
    averagesetall = []
    while w <= (len(nameset)-1):
        while j <= k - 1:      #find k nieghbors
            while i <= (len(outputset[0])-1): #find the nearst neighbor’s name in the Y_train.txt
                if nameset[w][j][1] == outputset[0][i]:    #if they have same name
                    target.append(i)  #Add the number of the name to the target list
                i = i + 1
            j = j + 1   #Traverse the second neighbor
            i = 0
        averageset.append(tf_test_all[w][0])
        w = w + 1
        j = 0
        while z <= (len(outputset) - 1):   #calculate the distance
            for x in target:
                sum = sum + float(outputset[z][x])
            average = sum/len(target)
            average = round(average,4)
            averageset.append(average)
            sum = 0
            average = 0
            z = z + 1
        averagesetall.append(averageset)
        averageset = []
        z = 1
    return averagesetall





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
def comparetf(tf_test_all,tf_train_all,points): # figure out the distance of testset and trainset
     distance = 0
     distanceset = []
     distancesetall = []
     namedistanceset = []
     i = 0
     j = 0

     while i<=(len(tf_test_all)-1):     #traversal all the X_train
         while j<=(len(tf_train_all)-1):     #traversal all the Y_unseen
            for x in points:      #set monitoring point
                 if tf_train_all[j][1][x] != tf_test_all[i][1][x]:   #Compare whether the test set and the training set are the same
                     distance = distance + 1
            namedistanceset.append(distance)     #Add the distance after the name
            namedistanceset.append(tf_train_all[j][0])
            j = j + 1
            distanceset.append(namedistanceset)
            distance = 0
            namedistanceset = []
         distanceset = sorted(distanceset, key=lambda x: x[0])
         distancesetall.append(distanceset)      #put one test factor distance in the set
         distanceset = []
         i = i + 1
         j = 0
     return distancesetall

"""
    Description:save the output as a file
    
    Parameters:
        filename - save the output in this file
        data - the predict output
    Returns:
        
"""

def text_save(filename, data):# save the output as a file
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)
    file.close()
    print("SAVED SUCCESSFUL! ")


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
    with open(observation_file) as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            observations.append(row)
    # print(observations)
    with open(test_file) as tsv:    #这个循环是用来提取测试集合的
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            test.append(row)
    print("==============================================")
    features, targets = prepare_data(observations)  #用来生成交叉验证用的features 和targets集合
    testfeatures = prepare_testdata(test)   #用来生成测试用的feature集合
    #print(features)
    #print(targets)
    '''
    outputset = outputsequence(file2)  #d
    #print(outputset)

    #c = comparetf(unseenset,trainset)   #find the distance of the neighbors
    #print(c)
    #k = input("please enter k for KNN algorithm: ")
    #e = classify(int(k),outputset,c,unseenset)
    #print(e)
    '''
    f = splitDataSet(features,targets,5)
    #print(f)
    '''
    print("==================")

    #g = denerageDataSet(f,0)
    #print(g)
    #print("==================")

    #h = performanceeachtrainset(5,g,outputset,[2,4,5,24,30,43,45,46,47])
    #print(h)
    #print("==================")

    #performancealltrainset(5,f,outputset,[2,4,5,24,30,43,45,46,47])
    #print(i)
    result = circileperformance()
    #print(result)
    output = open('model_selection_table.txt','w',encoding='gbk')
    output.write('\t\tD1Euclidean Distance\t\tD2Manhattan Distance\n')
    for i in range(len(result)):
        for j in range(len(result[i])):
            output.write(str(result[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
            output.write('\t')  # 相当于Tab一下，换一个单元格
        output.write('\n')  # 写完一行立马换行
    output.close()
    print("the result is in model_selection_table.txt")



    #colume_e = [[r[col] for r in e] for col in range(len(e[0]))]
    #print("The predict of the unseen data will be output as a file.(TF_predict.txt)")

    #text_save(filename3,colume_e)   #save the predict result as a file  predict.txt
    '''