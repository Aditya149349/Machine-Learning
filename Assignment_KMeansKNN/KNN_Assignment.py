import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math
import operator
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

class KNN:
    def __init__(self, metric='euclidean', k=6, p=3):
        self.metric = metric
        self.k=k
        self.p=p
        
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
        for k in range(1,16):
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

# Predicting without Pre-Processing the dataset, using euclidean distance measure and k=6

model = KNN()
predictions = model.knn(trainset_List, testset_list)
arr = model.check_best_k()
k_arr = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
plt.figure()
plt.plot(k_arr, arr)
plt.show()

# Pre-Processing the dataset using standard scalar

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

#shuffling the dataset
shuffle_df = df.sample(frac=1,random_state=10)
train_size = int(0.5*len(df))
#splitting data into train and test
train_set = shuffle_df[:train_size]
test_set = shuffle_df[train_size:]

trainset_List=train_set.values.tolist()
testset_list=test_set.values.tolist()

#predicting using euclidean distance measure and having k=6

model = KNN()
predictions = model.knn(trainset_List, testset_list)
arr = model.check_best_k()
k_arr=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
plt.figure()
plt.plot(k_arr, arr)
plt.title('K vs. Accuracy')
plt.show()

# Normalizing the data after it is Pre-Processed

df = pd.read_csv('diabetes.csv')
column_names = list(df.columns)
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
standardDf = standardDf.fillna(0)
normalized_df=preprocessing.normalize(standardDf)
normalized_df=pd.DataFrame(normalized_df,columns=column_names)
normalized_df['Outcome']=df['Outcome'].values
print("Pre-Processed data after normalization\n")
print(normalized_df)
df=normalized_df

#shuffling the dataset
shuffle_df = df.sample(frac=1,random_state=10)
train_size = int(0.5*len(df))
#splitting data into train and test
train_set = shuffle_df[:train_size]
test_set = shuffle_df[train_size:]

trainset_List=train_set.values.tolist()
testset_list=test_set.values.tolist()

#predicting using euclidean distance measure and having k=6
model = KNN()
predictions = model.knn(trainset_List, testset_list)
arr = model.check_best_k()
k_arr=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
plt.figure()
plt.plot(k_arr, arr)
plt.title('K vs. Accuracy')
plt.show()

# Changing Parameters(using Pre-Processed and normalized data)

#using manhattan distance instead of euclidean
model = KNN(metric = 'manhattan', k=5, p=2)
predictions = model.knn(trainset_List, testset_list)
arr = model.check_best_k()
k_arr=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
plt.figure()
plt.plot(k_arr, arr)
plt.title('K vs. Accuracy')
plt.show()

#using manhattan distance instead of euclidean
model = KNN(metric = 'minkowski', k=5, p=2)
predictions = model.knn(trainset_List, testset_list)
arr = model.check_best_k()
k_arr=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
plt.figure()
plt.plot(k_arr, arr)
plt.title('K vs. Accuracy')
plt.show()
