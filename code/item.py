'''
Author:Sheilaya Chong
Date:18/09/2018
Python Version:3.7
Data Source:https://github.com/revantkumar/Collaborative-Filtering(only part of the data is used here)
Paper:Item-Based Collaborative Filtering Recommendation Algorithms(http://files.grouplens.org/papers/www10_sarwar.pdf)

This program is a simple implementation of the above paper.
For similarity computation, I use cosine-based, correlation-based,and adjusted cosine similarity. And I compute all-to-all similarity.
I use weighted sum to make the prediction.

80%train+20%test, 5 fold cross validation.
MAE:0.8251(cosine-based), 0.8102(correlation-based), 0.4486（adjusted cosine similarity）
'''
import numpy as np
import scipy.stats
import scipy.spatial
from sklearn.model_selection import KFold
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
import math
import warnings
import sys
from numpy.linalg import norm



users = 1408
items = 2000

def readingFile(filename):
        f = open(filename,"r")
        data = []
        for row in f:
                r = row.split(',')
                #e=user item  rating
                e = [int(r[0]), int(r[1]), int(r[2])]
                data.append(e)
        return data

'''
#1.cosine-based
def similarity_item(data):
        print("beginsim")
        item_similarity_cosine = np.zeros((items,items))

        for item1 in range(items):
                print(item1)
                for item2 in range(items):
                        if item2>item1 and np.count_nonzero(data[:,item1]) and np.count_nonzero(data[:,item2]):

                                a=data[:,item1]
                                b=data[:,item2]
                                item_similarity_cosine[item1][item2] = np.inner(a, b)/(norm(a)*norm(b))
                         
                                       
        print("donesim")                
        return item_similarity_cosine'''

'''
#2.correlation-based
def similarity_item(data):
        print("beginsim")
        item_similarity_cosine = np.zeros((items,items))

        for item1 in range(items):
                print(item1)
                for item2 in range(items):
                        if item2>item1 and np.count_nonzero(data[:,item1]) and np.count_nonzero(data[:,item2]):
                                #average rating for item1,item2
                                r1=np.sum(data[:,item1])/np.count_nonzero(data[:,item1])
                                r2=np.sum(data[:,item2])/np.count_nonzero(data[:,item2])
                                #print(r1)
                                a=0
                                b=0
                                c=0
                                count=0
                                
                                for u in range(users):                                       
                                        if (data[u][item1]>0 and data[u][item2]>0):
                                                
                                                d1=data[u][item1]-r1
                                                d2=data[u][item2]-r2
                                               
                                                a=a+d1*d2
                                                b=b+d1*d1
                                                c=c+d2*d2
                                                count=count+1
                                                                                                           
                                if count>0 and b>0 and c>0:
                                        item_similarity_cosine[item1][item2] =a/(math.sqrt(b)*math.sqrt(c))
                                       
        print("donesim")                
        return item_similarity_cosine'''

#3.adjusted cosine
def similarity_item(data):
        print("beginsim")
        item_similarity_cosine = np.zeros((items,items))

        for item1 in range(items):
                print(item1)
                for item2 in range(items):
                        if item2>item1 and np.count_nonzero(data[:,item1]) and np.count_nonzero(data[:,item2]):
                                a=0
                                b=0
                                c=0
                                count=0
                                
                                for u in range(users):                                       
                                        if (data[u][item1]>0 and data[u][item2]>0):
                                                #average rating for user u
                                                ru=np.sum(data[u])/np.count_nonzero(data[u])
                                                d1=data[u][item1]-ru
                                                d2=data[u][item2]-ru
                                               
                                                a=a+d1*d2
                                                b=b+d1*d1
                                                c=c+d2*d2
                                                count=count+1
                                                                                                           
                                if count>0 and b>0 and c>0:
                                        item_similarity_cosine[item1][item2] =a/(math.sqrt(b)*math.sqrt(c))
                                       
        print("donesim")                
        return item_similarity_cosine

def crossValidation(data):
        #80%train,20%test
        k_fold = KFold(5,False,None)

        Mat = np.zeros((users,items))
        for e in data:
                Mat[e[0]-1][e[1]-1] = e[2]

        item_similarity_cosine = similarity_item(Mat)
        

        for train_indices, test_indices in k_fold.split(data):
                train = [data[i] for i in train_indices]
                test = [data[i] for i in test_indices]

                M = np.zeros((users,items))

                for e in train:
                        M[e[0]-1][e[1]-1] = e[2]

                true_rate = []
                pred_rate_cosine = []

                for e in test:
                        user = e[0]
                        item = e[1]

                        #item-based
                        if np.count_nonzero(M[user-1]):
                                a=0
                                b=0
                                predict=0
                                for n in range(items):
                                        if M[user-1][n]>0 and item_similarity_cosine[n][item-1]>0:
                                                print(user)
                                                print(item)
                                                print(n)
                                                print(item_similarity_cosine[n][item-1])
                                                print(M[user-1][n])
                                                a+=item_similarity_cosine[n][item-1]*M[user-1][n]
                                                b+=abs(item_similarity_cosine[n][item-1])

                                if b!=0:
                                        predict=a/b
                                true_rate.append(e[2])
                                pred_rate_cosine.append(predict)
                sum=0
                for i in range(len(true_rate)):
                        sum+=abs(true_rate[i]-pred_rate_cosine[i])
                        
                mae=sum/len(true_rate)
                print(len(true_rate))
                print(mae)
                print("donemae")
        return mae
recommend_data = readingFile("ratingsnew.csv")
crossValidation(recommend_data)


