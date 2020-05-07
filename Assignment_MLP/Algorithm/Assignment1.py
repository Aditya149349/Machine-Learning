import numpy as np
class Assignment1:
    def __init__(self, LRate=0.01, noIters=1000):
        self.lr=LRate
        self.noIters=noIters
        self.activation=self.stepFunc()
        self.weight= None
        self.bias=None

    def fit(self,x,y):
        n_rows, n_columns = x.shape

        self.weight= np.zeroes(n_columns)
        self.bias=0

        y_ = np.array([1 if i>0 else 0 for i in y])
        # loop for training the algorithm
        for _ in range(self.noIters):
            for idx, x_i in enumerate(x):
                linear_func= np.dot(x_i, self.weight)+self.bias
                y_pred=self.activation(linear_func)
                update = self.lr * (y_[idx]-y_pred)
                self.weight += update * x_i
                self.bias += update



    def predict(self,x):
        linear_func = np.dot(x, self.weights)+self.bias
        y_pred=self.activation(linear_func)
        return y_pred



    def stepFunc(self, x):
        return np.where(x>=0, 1, 0)



