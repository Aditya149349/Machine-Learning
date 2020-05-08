import numpy as np
class MLP:
    
    def __init__(self, learning_rate=0.01, n_iters=100, test_size=0.15):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.test_size = test_size
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None
        
    def split_dataset(self,X, y):
        '''
        Function to split the dataset into train and test data. Default size of test data is 15%.
        '''
        train_pct_index = int(self.test_size * len(X))
        X_test, X_train = X[:train_pct_index], X[train_pct_index:]
        y_test, y_train = y[:train_pct_index], y[train_pct_index:]
        return X_train, X_test, y_train, y_test

    def fit(self, X, y):
        '''
        Function to train the model on X and y training data points. Weights and Bias is initialized to 
        zero. This function prints the performance of the model for each iteration of training. Default 
        number of iterations is 100, while default learning rate is 0.01. 
        '''
        y_predicted = []
        X_train, X_test, y_train, y_test = self.split_dataset(X, y)
        
        n_samples, n_features = X_train.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array(y_train)

        for _ in range(self.n_iters):
            
            y_predicted = []
            for idx, x_i in enumerate(X_train):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted.append(self.activation_func(linear_output))
                
                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted[idx])

                self.weights += update * x_i
                self.bias += update
            print('Learned weights are: \n' + str(self.weights) + '\nHyperparameters: \nEpoch number: ' + str(_) + '  Learning Rate: ' + str(self.lr))
            self.print_accuracy_metrics(y_predicted, y_)

    def predict(self):
        '''
        This function predicts the values of X_test and returns the predictiions as a numpy array. 
        '''
        y_predicted = []
        X_train, X_test, y_train, y_test = self.split_dataset(X, y)
        linear_output = np.dot(X_test, self.weights) + self.bias
        y_predicted = (self.activation_func(linear_output))
        return y_predicted

    def _unit_step_func(self, x):
        '''
        The activation function used here is a unit step function. That means, if the input is 0 or above, 
        output is 1. While, if the input is negative, output is 0. 
        '''
        return np.where(x>=0, 1, 0)
        #return 1/(1 + np.exp(-x)) 
    
    def print_accuracy_metrics(self, y_predicted, y_test = []):
        '''
        This function prints the accuracy metrics of the model. Its being called for both training and 
        test data, so that accuracy can be measured as and when an iteration or epoch is completed. It 
        compares the true and predicted values and calculates accuracy, precision, error rate, and recall 
        along with the confusion matrix and prints them after each iteration.
        '''
        X_train, X_test, y_train, y_test = self.split_dataset(X, y)
        # error = sum(np.square(y_test - y_predicted))/len(y_test)
        TP, FP, TN, FN = self.performance_measure(y_test, y_predicted)
        accuracy = (TP + TN) / len(y_test)
        error = 1 - accuracy
        if TP + FP != 0:
            precision = TP / (TP+FP)
        else:
            precision = 'Non determinable'
        if TP + FN != 0:
            recall = TP/ (TP+FN)
        else:
            recall = 'Non determinable'
        confusion_matrix = [[TN, FP], [FN, TP]]
        print('Error: '+ str(error))
        print('Accuracy: '+ str(accuracy))
        print('Precision: '+ str(precision))
        print('Recall: '+ str(recall))
        print('Confusion Matrix: '+ str(confusion_matrix))
        return confusion_matrix
        
    def performance_measure(self, y_test, y_predicted):
        '''
        This function measures the True Positive, False Positive, True Negative, and False Negative 
        values and returns them. 
        '''
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        
        for i in range(0, len(y_test)): 
            if y_test[i]==y_predicted[i]==1:
               TP += 1
            if y_predicted[i]==1 and y_test[i]!=y_predicted[i]:
               FP += 1
            if y_test[i]==y_predicted[i]==0:
               TN += 1
            if y_predicted[i]==0 and y_test[i]!=y_predicted[i]:
               FN += 1
        return TP, FP, TN, FN


# Load the dataset - This dataset has 60 columns plus the decision variable 'R' and 'M'. It has a total of 208 rows.
X = np.loadtxt('sonar.txt', usecols=range(0,60), delimiter =',') 
labels = np.genfromtxt('sonar.txt', delimiter=',', usecols=-1, dtype=str)

# Traverse through the array Labels and replace 'R' with 1 and 'M' with 0. 
for i,label in enumerate(labels):
    if label == 'R':
        labels[i] = 1
    else:
        labels[i] = 0

y = np.array(labels)

# Shuffle the data so that the test data towards the end does not end up with only 0's
from mlxtend.preprocessing import shuffle_arrays_unison
X, y = shuffle_arrays_unison(arrays=[X, y], random_seed=3)

# Convert y from an array of strings to array of integers
y = y.astype(np.int)

# Initialize the MLP class and fit the multi-layer perceptron model on X and y data. X and y is split implicitly inside the 
# MLP class into training and test data
# Then predict the y values for X_test and store it in y_predicted
# The default number of iterations is 100, but can be changed by passing a value to n_iters. Similarly, learning rate is 
# 0.01, but can be changed. In each iteration, we are printing the accuracy metrics for the predicted values of the training
# data.
mlp = MLP()
mlp.fit(X,y)
y_predicted = mlp.predict()

# An accuracy of 70% has been gained on test data after 100 epochs. 
confusion_matrix = mlp.print_accuracy_metrics(y_predicted = y_predicted)

# CONFUSION MATRIX

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(confusion_matrix, range(2), range(2))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()
print('[[TN, FP], [FN, TP]]')
