import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm


# We have used three kernels linear, rbf and poly. We have also found the best C value among 0.001,0.01,0.1,1 and 10. We have also found the best gamma value among 0.1 and 1

# Load dataset
X = np.loadtxt('spambase.txt', usecols=range(0,57), delimiter =',') 
y = np.genfromtxt('spambase.txt', delimiter=',', usecols=-1, dtype=int)

# Convert X and y to numpy arrays
y = np.array(y)
X = np.array(X)

print(X.shape)
print(y.shape)

# Rearrange X and y 
from mlxtend.preprocessing import shuffle_arrays_unison

X, y = shuffle_arrays_unison(arrays=[X, y], random_seed=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Linear kernel

# When C = 0.001

linear_kernel = svm.SVC(C = 0.001, kernel='linear')
linear_kernel.fit(X_train,y_train)
linear_kernel.score(X_test,y_test)


# When C = 0.01

linear_kernel = svm.SVC(C = 0.01, kernel='linear')
linear_kernel.fit(X_train,y_train)
linear_kernel.score(X_test,y_test)

# When C = 0.1

linear_kernel = svm.SVC(C = 0.1, kernel='linear')
linear_kernel.fit(X_train,y_train)
linear_kernel.score(X_test,y_test)

# When C = 1

linear_kernel = svm.SVC(C = 1, kernel='linear')
linear_kernel.fit(X_train,y_train)
linear_kernel.score(X_test,y_test)


# Using GridSearchCV

# RBF kernel
# When C = 0.001

rbf_kernel = svm.SVC(C = 0.001, kernel='rbf')
rbf_kernel.fit(X_train,y_train)
rbf_kernel.score(X_test,y_test)

# When C = 0.01

rbf_kernel = svm.SVC(C = 0.01, kernel='rbf')
rbf_kernel.fit(X_train,y_train)
rbf_kernel.score(X_test,y_test)

# When C = 0.1

rbf_kernel = svm.SVC(C = 0.1, kernel='rbf')
rbf_kernel.fit(X_train,y_train)
rbf_kernel.score(X_test,y_test)

# When C = 1

rbf_kernel = svm.SVC(C = 1, kernel='rbf')
rbf_kernel.fit(X_train,y_train)
rbf_kernel.score(X_test,y_test)

# Polynomial kernel
# When C = 0.001

poly_kernel = svm.SVC(C = 0.001, kernel='poly')
poly_kernel.fit(X_train,y_train)
poly_kernel.score(X_test,y_test)

# When C = 0.01

poly_kernel = svm.SVC(C = 0.01, kernel='poly')
poly_kernel.fit(X_train,y_train)
poly_kernel.score(X_test,y_test)

# When C = 0.1

poly_kernel = svm.SVC(C = 0.1, kernel='poly')
poly_kernel.fit(X_train,y_train)
poly_kernel.score(X_test,y_test)

# When C = 1

poly_kernel = svm.SVC(C = 1, kernel='poly')
poly_kernel.fit(X_train,y_train)
poly_kernel.score(X_test,y_test)

# ## Using GridSearchCV

# ### Linear kernel

from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test=train_test_split(X[:100], y[:100], test_size=0.2)

grid_parameters={'C': [0.001,0.1,1,10], 'gamma': [0.001,0.0001], 'kernel':['linear']}
grid=GridSearchCV(svm.SVC(),grid_parameters,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_

# Run model with best parameters
linear_kernel = svm.SVC(C = 0.1, kernel='linear', gamma = 0.001)
linear_kernel.fit(X_train,y_train)
linear_kernel.score(X_test,y_test)

# ### RBF kernel
grid_parameters={'C': [0.001,0.1,1,10], 'gamma': [0.001,0.0001], 'kernel':['rbf']}
grid=GridSearchCV(svm.SVC(),grid_parameters,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_

# Run model with best parameters
rbf_kernel = svm.SVC(C = 1, kernel='rbf', gamma = 0.001)
rbf_kernel.fit(X_train,y_train)
rbf_kernel.score(X_test,y_test)

# ### Polynomial kernel
grid_parameters={'C': [0.001,0.1,1,10], 'gamma': [0.001,0.0001], 'kernel':['poly']}
grid=GridSearchCV(svm.SVC(),grid_parameters,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_

# Run model with best parameters
poly_kernel = svm.SVC(C = 10, gamma = 0.001, kernel='poly')
poly_kernel.fit(X_train,y_train)
poly_kernel.score(X_test,y_test)

# In conclusion, the linear kernel provides the best results. 
