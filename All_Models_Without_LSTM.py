#!/usr/bin/env python
# coding: utf-8

# # Single Day Stock Price Prediction Using Machine Learning Models

# In[1]:


# Importing all necessary libraries
import pandas as pd
import numpy as np


# In[2]:


# reading data from csv file
df = pd.read_csv('AXISBANK.csv')


# In[3]:


# printing the output of csv to check data
df.head()


# In[4]:


# importing all models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor


# In[5]:


# initialising all models
lin = LinearRegression()
rfr = RandomForestRegressor(n_estimators=100)
knn = KNeighborsRegressor(n_neighbors=3)
svr = SVR()
lin_svr = LinearSVR()
dt = DecisionTreeRegressor()


# In[6]:


# extracting important data from entire dataframe into new dataframe
new_df = df.drop(['Date','Symbol','Series','Prev Close','Last','VWAP','Volume','Turnover','Trades','Deliverable Volume','%Deliverble'],axis=1)
new_df.head()


# In[7]:


# importing test-train split to split the data into training and testing
from sklearn.model_selection import train_test_split


# In[8]:


X_train,X_test,y_train,y_test = train_test_split(new_df.drop('Open',axis=1),new_df['Open'],test_size=0.2,random_state=0)


# In[9]:


# Test_data = [[high,low,close]]
Test_data = [[729.85,705,714.90]]
# ans = 706


# # Linear Regression

# In[10]:


# fitting the model on train data
lin.fit(X_train,y_train)


# In[11]:


# finding the score to check whether model has been properly trained or not
lin.score(X_test,y_test)


# In[12]:


# predicting stock price by passing test data
prediction = lin.predict(Test_data)


# In[13]:


# actual output = 706
print('Actual Output - 706')
print('predicted output',prediction)


# # Random Forest Regressor

# In[14]:


rfr.fit(X_train,y_train)


# In[15]:


rfr.score(X_test,y_test)


# In[16]:


prediction = rfr.predict(Test_data)


# In[17]:


# output = 706
print(prediction)


# # K-Nearest Neighbour Regressor

# In[18]:


knn.fit(X_train,y_train)


# In[19]:


knn.score(X_test,y_test)


# In[20]:


prediction = knn.predict(Test_data)


# In[21]:


# output 706
print(prediction)


# # Support Vector Regressor

# In[22]:


svr.fit(X_train,y_train)


# In[23]:


svr.score(X_test,y_test)


# In[24]:


prediction = svr.predict(Test_data)


# In[25]:


# output = 706
print(prediction)


# # Linear Support Vector Regressor

# In[26]:


lin_svr.fit(X_train,y_train)


# In[27]:


lin_svr.score(X_test,y_test)


# In[28]:


prediction = lin_svr.predict(Test_data)


# In[29]:


# output = 706
print(prediction)


# # Decision Tree Regressor

# In[30]:


dt.fit(X_train,y_train)


# In[31]:


dt.score(X_test,y_test)


# In[32]:


prediction = dt.predict(Test_data)


# In[33]:


# output = 706
print(prediction)


# # Overall Result - Decision Tree Regressor seems to be the best model in case of predicting short term (i.e. single day) price prediction
