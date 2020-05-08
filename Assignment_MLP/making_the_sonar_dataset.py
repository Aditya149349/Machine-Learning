import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sonar = np.loadtxt('sonar.txt', usecols=range(0,10), delimiter =',') 
labels = np.genfromtxt('sonar.txt', delimiter=',', usecols=-1, dtype=str)

mlp = np.concatenate((sonar,labels[:,None]),axis=1)
mlp = pd.DataFrame({'A': mlp[:, 0], 'B': mlp[:, 1], 'C': mlp[:, 2], 'D': mlp[:, 3], 'E':mlp[:, 4], 'F':mlp[:, 5], 'G':mlp[:, 6], 'H':mlp[:, 7], 'I':mlp[:, 8], 'J':mlp[:, 9], 'Class': mlp[:, 10]})
mlp.Class = mlp.Class.map({'R': 0, 'M': 1})
mlp = mlp.sample(frac=1).reset_index(drop=True)
mlp.to_csv('sonar.csv', index=False)
