#!/usr/bin/env python
# coding: utf-8

# In[73]:


import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math
import operator
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


# In[74]:


class KNN:
    def __init__(self, metric='euclidean', k=6, p=3):
        self.metric = metric
        self.k=k
        self.p=p
 #euclidean distance       
    def straightLineDistance(self, row1,row2,length):
        distance=0
        for x in range(length):
            distance = distance+pow((row1[x]-row2[x]),2)
        return math.sqrt(distance)

    def p_root(self, value, root): 
        root_value = 1 / float(root) 
        return round (float(value) ** float(root_value), 3) 
  
    def minkowski_distance(self, x, y):   
        return (self.p_root(sum(pow(abs(a-b), self.p) for a, b in zip(x, y)), self.p))
    
    def manhattan_distance(self, x, y, n):
        sum = 0
        for i in range(n): 
            for j in range(i+1,n): 
                sum += (abs(x[i] - x[j]) + abs(y[i] - y[j])) 
        return sum
    
    def closestNeighbour(self, trainSet, testCase, k):
        distance=[]
        length = len(testCase)
        for x in range(len(trainSet)):
            if self.metric == 'euclidean':
                dist = self.straightLineDistance(testCase,trainSet[x], length)
            elif self.metric == 'minkowski':
                dist = self.minkowski_distance(testCase,trainSet[x])
            elif self.metric == 'manhattan':
                dist = self.manhattan_distance(testCase,trainSet[x], length)
            distance.append((trainSet[x],dist))
        distance.sort(key=operator.itemgetter(1))
        neighbour=[]
        for x in range(k):
            neighbour.append(distance[x][0])
        return neighbour
    
    def getResponse(self, neighbour):
        highestClass={}
        for x in range(len(neighbour)):
            labels=neighbour[x][-1]
            if labels in highestClass:
                highestClass[labels]= highestClass[labels]+1
            else:
                highestClass[labels]=1
        #sorting the nearest neighbours in descending order
        sortedLabels=sorted(highestClass.items(), key=operator.itemgetter(1), reverse=True)
        #returning the label which is most similar
        return sortedLabels[0][0]
    
    def knn(self, trainset_List, testset_list):
        predictions = []
        self.trainset_List = trainset_List
        self.testset_list = testset_list
        for x in range(len(trainset_List)):
            neighbors = self.closestNeighbour(trainset_List, testset_list[x], self.k)
            output = self.getResponse(neighbors)
            predictions.append(output)
#         return predictions
            print("predicted = ",output," expected = ",testset_list[x][-1])
        accepted=0;
        for x in range(len(testset_list)):
            if (testset_list[x][-1] ==  predictions[x]):
                accepted=accepted+1
        print("number of predictions which are correct :",accepted)
        accuracy= (accepted/float(len(testset_list))) * 100.0
        print("accuracy is ",accuracy)
        
    def check_best_k(self):
        arr = list()
        for k in range(1,21):
            predictions = []
            for x in range(len(trainset_List)):
                neighbors = self.closestNeighbour(trainset_List, testset_list[x], k)
                output = self.getResponse(neighbors)
                predictions.append(output)
    
            accepted = 0;
            for y in range(len(testset_list)):
                if (testset_list[y][-1] ==  predictions[y]):
                    accepted = accepted + 1
        
            accuracy = (accepted / float(len(testset_list))) * 100.0
            print("accuracy when k =",k,"is ",accuracy)
            arr.append(accuracy)
        print("Best k is :",arr.index(max(arr)) + 1)
        return arr
    
    


# # Shuffling and splitting the dataset into test and train data 

# In[75]:


df = pd.read_csv('diabetes.csv')
print("the diabetes dataset used consists of ",df.shape," number of rows and columns")

shuffle_df = df.sample(frac=1,random_state=10)
train_size=int(0.5*len(df))
print("After splitting the dataset: ")

train_set=shuffle_df[:train_size]
test_set=shuffle_df[train_size:]
print("training set :",len(train_set)," rows")
print("testing set :",len(test_set)," rows")

trainset_List=train_set.values.tolist()
testset_list=test_set.values.tolist()


# In[76]:


# Predicting without Pre-Processing the dataset, using euclidean distance measure and k=6


# In[77]:


model = KNN()
predictions = model.knn(trainset_List, testset_list)


# In[78]:


arr = model.check_best_k()


# In[79]:


k_arr = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.figure()
plt.plot(k_arr, arr)


# # Pre-Processing the dataset using standard scalar

# In[80]:


df = pd.read_csv('diabetes.csv')
column_names = list(df.columns)
del column_names[-1]
print("Data without Pre-Processing\n")
print(df)
print("\n")


std = StandardScaler()
standard = std.fit_transform(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']])
standardDf = pd.DataFrame(standard, columns = column_names)
standardDf['Outcome'] = df['Outcome'].values
df = standardDf
df['Pregnancies'] = df['Pregnancies'].abs()
df['Glucose'] = df['Glucose'].abs()
df['BloodPressure'] = df['BloodPressure'].abs()
df['SkinThickness'] = df['SkinThickness'].abs()
df['Insulin'] = df['Insulin'].abs()
df['BMI'] = df['BMI'].abs()
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].abs()
df['Age'] = df['Age'].abs()

print("after using standard scalar and converting the negatives to absolute values\n")

print(df)


# In[81]:


#shuffling the dataset
shuffle_df = df.sample(frac=1,random_state=10)
train_size = int(0.5*len(df))
#splitting data into train and test
train_set = shuffle_df[:train_size]
test_set = shuffle_df[train_size:]

trainset_List=train_set.values.tolist()
testset_list=test_set.values.tolist()


# In[82]:


#predicting using euclidean distance measure and having k=6
model = KNN()
predictions = model.knn(trainset_List, testset_list)


# In[83]:


arr = model.check_best_k()


# In[84]:


k_arr=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.figure()
plt.plot(k_arr, arr)


# # # Normalizing the train data after it is Pre-Processed

# In[85]:


df=pd.read_csv('diabetes.csv')
df2=pd.read_csv('diabetes.csv')
column_names=list(df.columns)

del column_names[-1]
std = StandardScaler()
standard = std.fit_transform(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']])
standardDf=pd.DataFrame(standard,columns=column_names)
standardDf['Pregnancies']=standardDf['Pregnancies'].abs()
standardDf['Glucose']=standardDf['Glucose'].abs()
standardDf['BloodPressure']=standardDf['BloodPressure'].abs()
standardDf['SkinThickness']=standardDf['SkinThickness'].abs()
standardDf['Insulin']=standardDf['Insulin'].abs()
standardDf['BMI']=standardDf['BMI'].abs()
standardDf['DiabetesPedigreeFunction']=standardDf['DiabetesPedigreeFunction'].abs()
standardDf['Age']=standardDf['Age'].abs()
standardDf['Outcome']=df['Outcome'].values
df=standardDf

#shuffling the dataset
shuffle_df=df.sample(frac=1,random_state=10)
train_size=int(0.5*len(df))
#splitting data into train and test
train_set=shuffle_df[:train_size]
test_set=shuffle_df[train_size:]
print("training set :",len(train_set)," rows")
print("testing set :",len(test_set)," rows\n")
ts=train_set
print("train set obtained after standard scaling the dataset\n")
print(train_set)
print("test set obtained after standard scaling the dataset\n")
print(test_set)


# # Normalizing only the training set 

# In[86]:


print("train_set after normalization\n")
train_set=preprocessing.normalize(train_set[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']])
train_set=pd.DataFrame(train_set,columns=column_names)
train_set['Outcome']=ts['Outcome'].values
print(train_set)


# In[87]:


trainset_List=train_set.values.tolist()
testset_list=test_set.values.tolist()


# In[88]:


#prediction for k=6 and euclidean distance measure


# In[89]:


model = KNN()
predictions = model.knn(trainset_List, testset_list)


# In[90]:


arr = model.check_best_k()


# In[91]:


k_arr = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.figure()
plt.plot(k_arr, arr)


# # Using different distance measure and different value for k

# In[101]:


#using the same test and train data since it has given the best accuracy


# In[93]:


#train data is standardized and normalized


# In[94]:


#test data is only standardized


# In[95]:


#using manhattan distance instead of euclidean
model = KNN(metric = 'manhattan', k=5, p=2)
predictions = model.knn(trainset_List, testset_list)


# In[96]:


arr = model.check_best_k()


# In[97]:


k_arr = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.figure()
plt.plot(k_arr, arr)


# In[69]:


#using minkowski distance instead of euclidean


# In[98]:


model = KNN(metric = 'minkowski', k=5, p=2)
predictions = model.knn(trainset_List, testset_list)


# In[99]:


arr = model.check_best_k()


# In[100]:


k_arr = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.figure()
plt.plot(k_arr, arr)


# In[ ]:




