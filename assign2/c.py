import sys
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=1e6)
from operator import itemgetter

"""
函数说明:打开输入文件，将文件存储为一个列表，切片得到每个元素

Parameters:
    filename - 文件名
Returns:
    tf - 以二维list形式返回输入输入的预测概率+真实分类，已经切片
Modify:
"""
def inputsequence(filename):

    with open(filename) as f:
        tf = []

        for line in f.readlines():
            tf_sq = line
            tf_sq_sp = tf_sq.split('\t')
            tf.append(tf_sq_sp)
            #print(a)
            #print(a)
            #print(type(a))
        #print(c)
        return tf

"""
函数说明:求出precision和recall的组合

Parameters:
    filename - 文件名
Returns:
    tf - 以二维list形式返回输入输入的预测概率+真实分类，已经切片
Modify:
"""
def calculate_pr(data):
    threshold = []
    a = 0
    k = 0
    precision_array = []
    recall_array = []
    for i in range(101):
        a = i/100
        threshold.append(a)
    #print(threshold)
    for j in threshold:
        #print("第几个阈值",j)
        TP = []    #把所有概率大于阈值且真实标记为1的存入
        FP = []    #把所有概率大于阈值且真实标记为0的存入
        FN = []    #把所有概率小于阈值且真实标记为1的存入
        #print("============",j,"=============")
        while k < len(data):
            #print("第几个阈值",j,"第几个元素",k)
            #print(data[k])
            #print(data[k][0])
            #print(j)
            #print(data[k][1])
            if float(data[k][0]) > j and int(data[k][1]) == 1:
                TP.append(data[k])
                #print(len(TP))
            if float(data[k][0]) > j and int(data[k][1]) == 0:
                FP.append(data[k])
            if float(data[k][0]) < j and int(data[k][1]) == 1:
                FN.append(data[k])
            k = k + 1
        if len(TP) == 0:
            precision = 0
            recall = 0
            precision_array.append(precision)
            recall_array.append(recall)
            continue
        #print(TP)
        #print(FP)
        #print(FN)
        precision = round(len(TP)/(len(TP)+len(FP)),3)
        recall = round(len(TP)/(len(TP)+len(FN)),3)
        #print(precision)
        #print(recall)
        precision_array.append(precision)
        recall_array.append(recall)
        k = 0
        #print(precision_array)
        #print(recall_array)

    return precision_array,recall_array,threshold


def text_save(filename, data):# save the output as a file
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)
    file.close()
    print("SAVED SUCCESSFUL! ")

def auprc_cal(precision_array,recall_array):
    i = 0
    area = 0
    area_sum = 0
    while i < (len(recall_array) - 1):
        print(precision_array[i])
        print(precision_array[i+1])
        print(recall_array[i+1]-recall_array[i])
        print("========================")
        area = ((precision_array[i] + precision_array[i+1]) * abs((recall_array[i+1]-recall_array[i]))) / 2
        print(area)
        print("========================")
        i = i + 1
        area_sum = area_sum + area
        area = 0
    return area_sum







"""
函数说明:main函数

Parameters:
    无
Returns:
    无

Modify:
    2017-03-24
"""
if __name__ == '__main__':
    #打开的文件名
    filename = sys.argv[1]   #"A2_T2_input.txt"
    file3 = "PR_table.txt"
    #打开并处理数据
    a = inputsequence(filename)
    #print(a)
    b = calculate_pr(a)
    #print("====================")
    #print(b)
    auprc = auprc_cal(b[0], b[1])
    print("AUPRC: ",auprc)
    plt.plot(b[0], b[1], 'r-o')
    plt.title("precision_recall_curve")
    plt.xlabel("recall")
    plt.ylabel("precision")
    #plt.text(0.5,0."AUPRC IS")
    #plt.show()
    plt.savefig("PRC.png")
    b[0].insert(0,"precision")
    b[1].insert(0,"recall")
    b[2].insert(0,"threshold")
    colume_e = [[r[col] for r in b] for col in range(len(b[0]))]
    #print(colume_e)
    text_save(file3, colume_e)
