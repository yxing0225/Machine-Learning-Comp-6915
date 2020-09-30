# -*- coding: UTF-8 -*-
import sys
import numpy as np
np.set_printoptions(threshold=1e6)
from operator import itemgetter

"""
Description:open a sequence and unseen file and store the file content as a list

Parameters:
    filename - the name of the file 
Returns:
    tf - return the content of file as a list,and use tab to split
"""
def inputsequence(filename):  # open the input file and store the file content as a list. X_train.txt and X.unseen.txt will use this function

    with open(filename) as f:
        tf = []

        for line in f.readlines():
            tf_sq = line
            tf_sq_sp = tf_sq.split('\t')
            tf.append(tf_sq_sp)
        return tf

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
def comparetf(tf_test_all,tf_train_all): # figure out the distance of testset and trainset
     distance = 0
     distanceset = []
     distancesetall = []
     namedistanceset = []
     i = 0
     j = 0

     while i<=(len(tf_test_all)-1):     #traversal all the X_train
         while j<=(len(tf_train_all)-1):     #traversal all the Y_unseen
            for x in [2,4,5,24,30,43,45,46,47]:      #set monitoring point
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
    #open files
    file = sys.argv[1]  #"X_train.txt"
    file1 = sys.argv[3]  #"toTest.txt"
    file2 = sys.argv[2]  #"Y_train.txt"
    file3 = "TF_predict.txt"

    #transfer the content of file to list and store in variable
    trainset = inputsequence(file)
    unseenset = inputsequence(file1)
    outputset = outputsequence(file2)

    c = comparetf(unseenset,trainset)   #find the distance of the neighbors
    #print(c)
    k = input("please enter k for KNN algorithm: ")
    e = classify(int(k),outputset,c,unseenset)
    #print(e)
    colume_e = [[r[col] for r in e] for col in range(len(e[0]))]
    print("The predict of the unseen data will be output as a file.(TF_predict.txt)")

    text_save(file3,colume_e)   #save the predict result as a file  predict.txt