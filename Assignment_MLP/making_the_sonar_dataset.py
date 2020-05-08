#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


sonar = np.loadtxt('sonar.txt', usecols=range(0,10), delimiter =',') 


# In[3]:


sonar.shape


# In[4]:


y = np.genfromtxt('sonar.txt',dtype='str')


# In[5]:


labels = np.genfromtxt('sonar.txt', delimiter=',', usecols=-1, dtype=str)


# In[6]:


labels


# In[7]:


mlp = np.concatenate((sonar,labels[:,None]),axis=1)


# In[8]:


import pandas as pd
mlp = pd.DataFrame({'A': mlp[:, 0], 'B': mlp[:, 1], 'C': mlp[:, 2], 'D': mlp[:, 3], 'E':mlp[:, 4], 'F':mlp[:, 5], 'G':mlp[:, 6], 'H':mlp[:, 7], 'I':mlp[:, 8], 'J':mlp[:, 9], 'Class': mlp[:, 10]})


# In[9]:


mlp.head()


# In[10]:


mlp.Class = mlp.Class.map({'R': 0, 'M': 1})


# In[11]:


mlp.head()


# In[12]:


mlp = mlp.sample(frac=1).reset_index(drop=True)


# In[13]:


mlp.to_csv('sonar.csv', index=False)


# In[14]:


mlp.describe()


# In[15]:


mlp.head(30)

