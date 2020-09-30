import sys
import csv
import numpy as np      #有用
import pandas as pd    #有用
np.set_printoptions(threshold=1e6)
from numpy import *
from sklearn import datasets
#from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier   #有用
from operator import itemgetter
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score   #有用

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
    #print(targets)

    return features, targets

def prepare_testdata(test_data):
    train_data = array(test_data)
    features = train_data[:, :]
    #print(shape(train_data))
    testfeatures = []
    transfer = []
    for i in features[1:]:
        for j in i:
            j = float(j)
            transfer.append(j)
        testfeatures.append(transfer)
        transfer = []
    return testfeatures

def transferfeatures(features):    #把字符串型变为float类型，把feature值转化为[[0.0, 0.0, 1.0, 0.051948052, 0.025974026, 0.077922078],
    transfereach = []  # transfer the word to float
    transferall = []
    for i in features[1:]:
        for j in i:
            j = float(j)
            transfereach.append(j)
        transferall.append(transfereach)
        transfereach = []
    return transferall

def transfertargets(targets):    #把字符串型变为float类型，把targets值转化为[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,...]格式
    transfereach = []  # transfer the word to float
    #transferall = []
    for i in targets[1:]:
        for j in i:
            j = float(j)
            transfereach.append(j)
        #transferall.append(transfereach)
        #transfereach = []
    return transfereach


def KNN(features,targets):
    k_range = range(4, 7)
    cv_score = []
    for k in k_range:
        knn = KNeighborsClassifier(k)
        scores = cross_val_score(knn, features, targets, cv=10, scoring="accuracy")
        score_mean = scores.mean()
        #score_std = scores.std()
        cv_score.append(score_mean)
        print(k, score_mean)
        #print("standard devision:")
        #print(score_std)
    best_k = np.argmax(cv_score) + 1
    #print("最优的k是%i" % (best_k))
    #plt.plot(k_range, cv_score)
    #plt.xlabel("k")
    #plt.ylabel("score")
    #plt.show()
    return best_k, max(cv_score)

def LOGICR(features,targets):
    featurenum_range = [10,30,60]
    cv_score = []
    for num in featurenum_range:
        transfereach = []  # transfer the word to float
        transferall = []
        count = 0
        for i in features:
            for j in i:
                if count < num:
                    transfereach.append(j)
                    count = count + 1
            transferall.append(transfereach)
            transfereach = []
            count = 0
        logicr = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression())])
        scores = cross_val_score(logicr, transferall, targets, cv=10, scoring="accuracy")
        score_mean = scores.mean()
        #score_std = scores.std()
        cv_score.append(score_mean)
        print(num, score_mean)
        #print("standard devision:")
        #print(score_std)
    best_num = np.argmax(cv_score) + 1
    #print("最优的featurenum是%i" % (best_num))
    #best_score = max(cv_score)
    #print("最优的SCORE是%i" % (best_score))
    #plt.plot(featurenum_range, cv_score)
    #plt.xlabel("number of features")
    #plt.ylabel("score")
    #plt.show()
    return best_num, max(cv_score)


def TREE(features,targets):
    depth_range = range(3, 8)
    cv_score = []
    for depth in depth_range:
        tre = tree.DecisionTreeClassifier(max_depth=depth)
        scores = cross_val_score(tre, features, targets, cv=10, scoring="accuracy")
        score_mean = scores.mean()
        #score_std = scores.std()
        cv_score.append(score_mean)
        print(depth, score_mean)
        #print("standard devision:")
        #print(score_std)
    best_depth = np.argmax(cv_score) + 1
    #print("最优的depth是%i" % (best_depth))
    best_score = max(cv_score)
    #print("最优的SCORE是%i" % (best_score))
    #plt.plot(depth_range, cv_score)
    #plt.xlabel("tree's max depth")
    #plt.ylabel("score")
    #plt.show()
    return best_depth, max(cv_score)

def bestmodel(features, targets, testfeatures):
    score = {}
    knn = KNN(features, targets)
    score['1'] = knn[1]
    print("==============================================")
    logicr = LOGICR(features, targets)
    score['2'] = logicr[1]
    print("==============================================")
    tree1 = TREE(features, targets)
    score['3'] = tree1[1]
    print("==============================================")
    orderscore = sorted(score.items(), key=lambda x: x[1],reverse = True)
    print(orderscore)
    if orderscore[1][0] == '1':
        print("the best method is KNN.")
        knn = KNeighborsClassifier(knn[0])
        knn.fit(features,targets)
        #print(knn.predict(testfeatures))
        predict = knn.predict_proba(testfeatures)
        return predict
    if orderscore[1][0] == '2':
        print("the best method is LOGIC REGRESSION.")
        logicrfinal = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression())])
        transfereach = []
        transferall = []
        count = 0
        for i in features:
            for j in i:
                if count <= logicr[0]:
                    transfereach.append(j)
                    count = count + 1
            transferall.append(transfereach)
            transfereach = []
            count = 0
        logicrfinal.fit(transferall, targets)
        '''
        =============================
        '''
        testtransfereach = []
        testtransferall = []
        count = 0
        for i in testfeatures:
            for j in i:
                if count <= logicr[0]:
                    testtransfereach.append(j)
                    count = count + 1
            testtransferall.append(testtransfereach)
            testtransfereach = []
            count = 0
        #print("test数据的shape：",shape(testtransferall))

        predict = logicrfinal.predict_proba(testtransferall)
        return predict
    if orderscore[1][0] == '3':
        print("the best method is DECISION TREE.")
        treefinal = tree.DecisionTreeClassifier(max_depth=tree1[0])
        treefinal.fit(features,targets)
        predict = treefinal.predict_proba(testfeatures)
        return predict

def text_save(filename, data): #filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("\nThe predictions are in g9_predictions.txt ")





if __name__ == '__main__':
    #from ipdb import set_trace
    print("Wait a minute please, it will takes about half minute to process ......")
    filename3 = "g9_predictions.txt"

    observation_file = sys.argv[1]       #"A3_training_dataset.tsv"
    test_file =  sys.argv[2]     #"A3_test_dataset.tsv"
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

        #print(observations)
        #print(test)
        #print("==============================================")
        features, targets = prepare_data(observations)  # 用来生成交叉验证用的features 和targets集合
        testfeatures = prepare_testdata(test)  # 用来生成测试用的feature集合
        #print("testdata_features:", testfeatures)
        #print("features",features)
        #print("targets",targets)
        #print("==============================================")
        features = transferfeatures(features)  #把feature值转化为[[0.0, 0.0, 1.0, 0.051948052, 0.025974026, 0.077922078], [1.0, 1.0, 1.0, 0.026239067, 0.025614327, 0.05997501],...]格式
        targets = transfertargets(targets)  #把targets值转化为[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,...]格式
        #print("==============================================")
        #KNN(features,targets)
        #print("==============================================")
        #LOGICR(features,targets)
        #print("==============================================")
        #TREE(features,targets)
        #print("==============================================")
        #print(TREE(features,targets))
        #print("==============================================")
        final_prob = bestmodel(features,targets,testfeatures[1:])
        #print(final_prob)
        final_prob_1 = []
        for i in final_prob:
            final_prob_1.append(i[1])
        #print(final_prob_1)
        final_prob_1.insert(0,"the prediction of likelihood to belongs to class 1 are as below:")

        final_prob_1 = np.array(final_prob_1)

        final_prob_1.reshape(final_prob_1.shape[0], 1)

        #colume_final_prob_1 = [[r[col] for r in final_prob_1] for col in range(len(final_prob_1))]
        #print("The predict of the unseen data will be output as a file.(gN.txt)")

        text_save(filename3, final_prob_1)  # save the predict result as a file  predict.txt

