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
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('Classification.CancerMB.csv')


# In[3]:


df.head()


# In[4]:


df = df.drop({'Unnamed: 32','id'},axis=1)


# In[5]:


df['diagnosis'] = df['diagnosis'].map({'M':0,'B':1})


# In[6]:


x = df.drop('diagnosis',axis=1)
y = df['diagnosis']


# In[7]:


#Splitting dataset into train test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)

def Master(model, model_name):
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    acuracy = accuracy_score(y_test, y_pred)
    print(model_name, " : ", acuracy)


# In[8]:


#popeline
classifier = KNeighborsClassifier(n_neighbors = 5)

pipe = [('standard_Scaler',StandardScaler()),('classifier',classifier)]
pipe = Pipeline(pipe)


# In[9]:


#Calling our master function
Master(pipe,'KNN')


# In[11]:


param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
regressor = classifier


# In[12]:


from sklearn.model_selection import GridSearchCV
regressorcv =  GridSearchCV(classifier,param_grid,cv = 5)


# In[13]:


regressorcv.fit(x_train,y_train)


# In[14]:


y_pred = regressorcv.predict(x_test)


# In[15]:


accuracy_score(y_test,y_pred)


# In[16]:


regressorcv.best_params_


# In[17]:


steps = [('standardscaler',StandardScaler()),('classifier',LogisticRegression())]


# In[19]:


pipe2 = Pipeline(steps)


# In[20]:


#Calling our master function
Master(pipe2,'LogisticRegression')


# In[23]:


#popeline
classifier2 = LogisticRegression()

pipe2 = [('standard_Scaler',StandardScaler()),('classifier',classifier2)]
pipe2 = Pipeline(pipe2)


# In[24]:


Master(pipe2,'LogisticRegression')


# In[25]:


param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
regressor = classifier2


# In[26]:


regressorcv.fit(x_train,y_train)


# In[27]:


y_pred = regressorcv.predict(x_test)


# In[28]:


accuracy_score(y_test,y_pred)


# In[ ]:




