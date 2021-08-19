#!/usr/bin/env python
# coding: utf-8

# ### Stock Price Prediction Using LSTM

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('AXISBANK.csv')


# In[3]:


df.head()


# In[4]:


# extracting data from column "Close" into new dataframe
df1=df.reset_index()['Close']


# In[5]:


plt.plot(df1)


# In[6]:


# LSTM are sensitive to the scale of the data. so we apply MinMax scaler to scale the values in between 0 and 1
from sklearn.preprocessing import MinMaxScaler
# To scale the values in between 0 and 1
sc = MinMaxScaler(feature_range=(0,1)) 
df1=sc.fit_transform(np.array(df1).reshape(-1,1))


# In[7]:


print(df1)


# In[8]:


#splitting dataset into train and test split
train_size=int(len(df1)*0.65)
test_size=len(df1)-train_size
train_data,test_data=df1[0:train_size,:],df1[train_size:len(df1),:1]


# In[9]:


def to_update_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[10]:


time_step = 100
X_train, y_train = to_update_dataset(train_data, time_step)
X_test, ytest = to_update_dataset(test_data, time_step)


# In[11]:


# reshape input into such a manner that it can be taken by LSTM [samples, time steps, features]
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[12]:


# Creating the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[13]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[14]:


model.summary()


# In[15]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=5,batch_size=64,verbose=1)


# In[16]:


import tensorflow as tf


# In[17]:


# Prediction
train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)


# In[18]:


# Transform output back to original form so it can be compared with actual values
train_prediction=sc.inverse_transform(train_prediction)
test_prediction=sc.inverse_transform(test_prediction)


# In[19]:


x_input=test_data[341:].reshape(1,-1)


# In[20]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[21]:


# Calculating prediction for next 30 days
lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        y_predicted = model.predict(x_input, verbose=0)
        temp_input.extend(y_predicted[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(y_predicted.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        y_predicted = model.predict(x_input, verbose=0)
#         print(yhat[0])
        temp_input.extend(y_predicted[0].tolist())
#         print(len(temp_input))
        lst_output.extend(y_predicted.tolist())
        i=i+1
    
print('Output of all days from day 1 to day 30')
print(lst_output)


# In[22]:


d_new=np.arange(1,101)
d_pred=np.arange(101,131)


# In[23]:


# plotting our predicted output from day 1 t day 30 using orange line
plt.plot(d_new,sc.inverse_transform(df1[1158:]))
plt.plot(d_pred,sc.inverse_transform(lst_output))


# In[24]:


# plotting actual output from the data that we have 
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])

