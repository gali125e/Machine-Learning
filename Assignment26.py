#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.metrics import accuracy_score,mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('Classification.CancerMB.csv')


# In[4]:


df.head()


# In[5]:


df.isnull().sum()


# In[6]:


df.shape


# In[7]:


df['diagnosis'].unique()


# In[8]:


df = df.drop({'Unnamed: 32','id'},axis=1)


# In[9]:


df.head()


# In[10]:


df['diagnosis'] = df['diagnosis'].map({'M':0,'B':1})


# In[11]:


plt.figure(figsize=(30,15))
sns.heatmap(df.corr(), center=0, cmap='BrBG', annot=True)


# ##It seems like dignosis has very small correlation with other parameters but they have strong correlation with each other so we can not directly drop any of the features 

# In[12]:


x = df.drop('diagnosis',axis=1)
y = df['diagnosis']


# In[13]:


from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.1)
print(selector.fit_transform(x))


# In[14]:


print(selector.get_support(indices=True))


# In[15]:


num_cols = list(x.columns[selector.get_support(indices=True)])

print(num_cols)


# In[16]:


x_t = x[num_cols]

x_t


# In[17]:


#Splitting dataset into train test
x_train,x_test,y_train,y_test = train_test_split(x_t,y,test_size = 0.2, random_state = 1)

def Master(model, model_name):
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    acuracy = accuracy_score(y_test, y_pred)
    
    print(model_name, "Accuracy : ", acuracy)
    print(classification_report(y_test,y_pred),'auc',roc_auc_score(y_test,y_pred))


# In[18]:


#popeline
classifier = KNeighborsClassifier(n_neighbors = 5)

pipe = [('standard_Scaler',StandardScaler()),('classifier',classifier)]
pipe = Pipeline(pipe)


# In[19]:


#Calling our master function
Master(pipe,'KNN')


# In[20]:


steps = [('standardscaler',StandardScaler()),('classifier',LogisticRegression())]


# In[21]:


pipe2 = Pipeline(steps)


# In[22]:


#Calling our master function
Master(pipe2,'LogisticRegression')


# # For Wrapper Function

# In[23]:


get_ipython().system('pip install mlxtend')


# In[24]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression


# In[25]:


lreg = LinearRegression()
sfs1 = sfs(lreg, k_features=14, forward=True, verbose=2, scoring='neg_mean_squared_error')


# In[26]:


sfs1 = sfs1.fit(x, y)


# In[27]:


feat_names = list(sfs1.k_feature_names_)
feat_names


# In[28]:


new_data = x[feat_names]

# first five rows of the new data
new_data.head()


# In[29]:


new_data.shape, x.shape


# In[30]:


x_train,x_test,y_train,y_test = train_test_split(new_data,y,test_size = 0.2, random_state = 1)


# In[31]:


#Calling our master function
Master(pipe,'KNN')


# In[32]:


#Calling our master function
Master(pipe2,'LogisticRegression')


# In[33]:


lreg = LinearRegression()
sfs1 = sfs(lreg, k_features=14, forward=False, verbose=2, scoring='neg_mean_squared_error')


# In[34]:


sfs1 = sfs1.fit(x, y)


# In[35]:


feat_names_back = list(sfs1.k_feature_names_)
feat_names_back


# In[36]:


new_data_back = x[feat_names_back]

# first five rows of the new data
new_data_back.head()


# In[37]:


new_data.shape, x.shape


# In[38]:


x_train,x_test,y_train,y_test = train_test_split(new_data_back,y,test_size = 0.2, random_state = 1)


# In[39]:


#Calling our master function
Master(pipe,'KNN')


# In[40]:


#Calling our master function
Master(pipe2,'LogisticRegression')


# In[41]:


# Importing PCA
from sklearn.decomposition import PCA

# Let's say, 
n_components = 2
pca = PCA(n_components=n_components)
pca.fit(x)
x_pca = pca.transform(x)

# Create the dataframe
df_pca1 = pd.DataFrame(x_pca,
					columns=['PC{}'.
					format(i+1)
						for i in range(n_components)])
print(df_pca1)


# In[42]:


x_train,x_test,y_train,y_test = train_test_split(df_pca1,y,test_size = 0.2, random_state = 1)


# In[43]:


#Calling our master function
Master(pipe,'KNN')


# In[44]:


#Calling our master function
Master(pipe2,'LogisticRegression')


# In[45]:


df =pd.read_csv('Regression.Life.Expectancy.csv')


# In[46]:


df.head()


# In[47]:


df.isnull().sum()


# In[ ]:





# In[48]:


df.isnull().sum()


# In[49]:


df


# In[50]:


df['Country'].unique()


# In[51]:


labelencoder = LabelEncoder()
df['Country'] = labelencoder.fit_transform(df['Country'])
df['Status'] = labelencoder.fit_transform(df['Status'])


# In[52]:


df.mean()


# In[53]:


df = df.fillna(df.mean())


# In[54]:


df.isnull().sum()


# In[55]:


df.head()


# In[56]:


x_reg = df.drop('Life expectancy ',axis = 1)


# In[57]:


x_reg.head()


# In[58]:


y = df['Life expectancy ']


# In[59]:


y.head()


# In[60]:


plt.figure(figsize=(30,15))
sns.heatmap(df.corr(), center=0, cmap='BrBG', annot=True)


# ##As one can see that Life expectancy has almost zero correlation with country ,measeles and population so it will be better to remove these
# two features

# In[61]:


x_reg = x_reg.drop(['Population','Country','Measles '],axis=1)


# In[62]:


x_reg.head()


# In[63]:


from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
print(selector.fit_transform(x_reg))


# In[64]:


print(selector.get_support(indices=True))


# In[65]:


num_cols = list(x_reg.columns[selector.get_support(indices=True)])

print(num_cols)


# In[66]:


x_reg = x_reg[num_cols]

x_reg


# In[67]:


#Splitting dataset into train test




def Master2(model, model_name):
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    acuracy = r2_score(y_test, y_pred)
    #acuracy = accuracy_score(y_test, y_pred)
    
    print(model_name, "Accuracy : ", acuracy)
    print(mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))
    #print(classification_report(y_test,y_pred),'auc',roc_auc_score(y_test,y_pred))

    #print(model_name, " : ", acuracy)


# In[68]:


steps1 = [('standardscaler',StandardScaler()),('classifier',LinearRegression())]


# In[69]:


pipe3 = Pipeline(steps1)


# In[70]:


x_train,x_test,y_train,y_test = train_test_split(x_reg,y,test_size = 0.2, random_state = 1)


# In[71]:


Master2(pipe3,'LinearRegression')


# In[72]:


from sklearn.linear_model import ElasticNet


# In[73]:


steps2 = [('standardscaler',StandardScaler()),('classifier',ElasticNet())]


# In[74]:


pipe4 = Pipeline(steps2)


# In[75]:


Master2(pipe4,'ElasticNet')


# In[76]:


## Wrapper Functions


# In[77]:


sfs1 = sfs1.fit(x_reg, y)


# In[78]:


feat_names_back_reg = list(sfs1.k_feature_names_)
feat_names_back_reg


# In[79]:


new_data_back_reg = x_reg[feat_names_back_reg]

# first five rows of the new data
new_data_back_reg.head()


# In[80]:


new_data_back_reg.shape, x_reg.shape


# In[81]:


x_train,x_test,y_train,y_test = train_test_split(new_data_back_reg,y,test_size = 0.2, random_state = 1)


# In[82]:


Master2(pipe3,'LinearRegression')


# In[83]:


Master2(pipe4,'ElasticNet')


# In[84]:


# Let's say, 
n_components = 6
pca = PCA(n_components=n_components)
pca.fit(x_reg)
x_pca_reg = pca.transform(x_reg)

# Create the dataframe
df_pca1_reg = pd.DataFrame(x_pca_reg,
					columns=['PC{}'.
					format(i+1)
						for i in range(n_components)])
print(df_pca1_reg)



# In[85]:


x_train,x_test,y_train,y_test = train_test_split(df_pca1_reg,y,test_size = 0.2, random_state = 1)


# In[86]:


Master2(pipe3,'LinearRegression')


# In[87]:


Master2(pipe4,'ElasticNet')


# In[ ]:




