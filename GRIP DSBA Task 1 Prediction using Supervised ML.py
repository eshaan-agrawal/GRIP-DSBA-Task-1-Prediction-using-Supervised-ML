#!/usr/bin/env python
# coding: utf-8

# # Task 1: Prediction using Supervised ML

# ### Author: Eshaan Agrawal

# ### Let's begin 

# Importing the essential libraries for the task.

# In[1]:


import pandas as pd, numpy as np, matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importing the data

# Importing the data from the given url: https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv

# In[2]:


link_to_data = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
raw_data = pd.read_csv(link_to_data)
print("Data Imported")


# Checking the data.

# In[3]:


raw_data


# ### Understanding the data

# Understanding any preexisting relations in the data.

# In[4]:


raw_data.plot(x='Hours', y='Scores', style='s')  
plt.title('Hours vs Scores')  
plt.xlabel('Hours Studied')  
plt.ylabel('Scores');  


# In[5]:


raw_data.corr()


# A strong positive linear relation is seen between Hours and Scores.

# ### Preparing the data

# Splitting our data into numpy arrays to move forward.

# In[6]:


X = raw_data.iloc[:,:-1].values
y = raw_data.iloc[:,1].values


# Here, `X` is required to be a 2D array.

# ### Splitting the data

# Importing the `train_test_split` method from the `sklearn.model_selection`

# In[7]:


from sklearn.model_selection import train_test_split


# Splitting our data into train and test datasets. We'll be training our model with 80% of the available data.

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# Checking if everything is well according to our plan.

# In[9]:


X_test


# ### Algorithm

# Finally importing `LinearRegression` from `sklearn.linear_model`

# In[10]:


from sklearn.linear_model import LinearRegression


# Making an object named `reg` to move forward.

# In[11]:


reg = LinearRegression()


# ### Training our model

# Training our model with the training data using the `fit` attribute.

# In[12]:


reg.fit(X_train,y_train)


# In[13]:


# Plotting the regression line
reg_line = reg.coef_*X+reg.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.title("Regression line and Test data")
plt.plot(X, reg_line);
plt.show()


# ### Predicting using the trained model

# Predicting on `X_test` data using our trained model.

# In[14]:


y_predicted = reg.predict(X_test)


# In[15]:


y_predicted


# ### Comparing

# Comparing the actual and the predicted scores.

# In[16]:


df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_predicted})  
df_compare


# ### Predicting for new data

# Predicting the score for some new data.

# In[17]:


hours = [[9.25]] #as input must a 2D array always
predicted_score = reg.predict(hours)
print("No of Hours = {}".format(hours[0][0]))
print("Predicted Score = {}".format(predicted_score[0]))


# ### Evaluating the model

# Finally we'll be evaluating the model, so that we can compare it with other models.
# 
# We have chosen the mean square error metric for evaluation here, there are many such metrics.

# In[18]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predicted)) 

