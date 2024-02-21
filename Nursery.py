#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv('nursery.data')


# In[6]:


df


# In[14]:


df = df.drop('recommend',axis = 1)


# In[15]:


df.columns


# In[16]:


labelencoder = LabelEncoder()
df['usual'] = labelencoder.fit_transform(df['usual'])
df['proper'] = labelencoder.fit_transform(df['proper'])
df['complete'] = labelencoder.fit_transform(df['complete'])
df['1'] = labelencoder.fit_transform(df['1'])
df['convenient'] = labelencoder.fit_transform(df['convenient'])
df['convenient.1'] = labelencoder.fit_transform(df['convenient.1'])
df['nonprob'] = labelencoder.fit_transform(df['nonprob'])
df['recommended'] = labelencoder.fit_transform(df['recommended'])


# In[17]:


df


# In[18]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), center=0, cmap='BrBG', annot=True)


# #Since the data was most categorical so they are not very much correlated

# In[21]:


X = df.drop('recommended',axis = 1)


# In[22]:


X


# In[23]:


Y = df['recommended']


# In[24]:


Y


# In[25]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2, random_state = 1)

def Master(model, model_name):
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    acuracy = accuracy_score(y_test, y_pred)
    print(model_name, " : ", acuracy)


# In[26]:


classifier = KNeighborsClassifier(n_neighbors = 5)

pipe = [('standard_Scaler',StandardScaler()),('classifier',classifier)]
pipe = Pipeline(pipe)


# In[27]:


Master(pipe,'KNN')


# In[28]:


param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}


# In[29]:


cv =  GridSearchCV(classifier,param_grid,cv = 5)
cv.fit(x_train,y_train)


# In[30]:


y_pred =cv.predict(x_test)


# In[31]:


accuracy_score(y_test,y_pred)


# In[32]:


cv.best_params_


# In[ ]:




