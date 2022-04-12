#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Name: Ishan Vyas
# UCID: 30068270
# Course: CPSC 502.06
# Project: Regression testing for machine learning classifiers
# Supervisor: Dr. Frank Maurer

# Sample Program based around the 'diabetes' dataset.


# In[2]:


# This is a simple diabetes predictor program that predicts 
# if a person has diabetes or not based on a KNN machine learning model.
# Database link: https://www.kaggle.com/datasets/mathchi/diabetes-data-set


# In[3]:


# Import statements
from hashdf import HashableDataFrame

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[4]:

# This function loads the dataset
def load_dataset(fileName):
        return pd.read_csv(fileName)


# In[5]:

# This function is responsible for finding the best K value for the given dataset
def find_bestK(start, end, df):
    data = df.loc[:, df.columns != 'Outcome']
    labels = df.loc[:, df.columns == 'Outcome']
    
    k_value_range = range(start,end)
    # Result scores
    k_value_scores = []
    K = 0
    for k in k_value_range:
        knn_model = KNeighborsClassifier(n_neighbors = k)
        accuracy = cross_val_score(knn_model, data, labels.values.ravel(), cv=10, scoring="accuracy")
        k_value_scores.append(accuracy.mean())

    K = k_value_scores.index(max(k_value_scores))+1
    return K


# In[6]:

# This function is responsible for training a new classifier based on the provided dataset
def getModel(db, K):
    data = db.loc[:, db.columns != 'Outcome']
    labels = db.loc[:, db.columns == 'Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels.values.ravel(), test_size=0.25, random_state=1)

    neigh = KNeighborsClassifier(n_neighbors=K)  
    neigh.fit(X_train, y_train)

    Initial = neigh.predict(X_test)
    return neigh, Initial


# In[7]:


# Load the diabetes dataset
df = load_dataset('diabetes.csv')
# get_ipython().run_line_magic('store', 'df')


# In[8]:


# Show database information
print(df.info())
# Show the first 5 rows of the dataset
print(df.head())


# In[9]:


# Change the current dataframe into a hashable one so that we can pass it as a parameter 
df_hash = HashableDataFrame(df)
# get_ipython().run_line_magic('store', 'df_hash')


# In[10]:


# Find the best K value
K = find_bestK(1,30, df_hash)
# get_ipython().run_line_magic('store', 'K')


# In[11]:

# Get the classifer for the given dataset
diabetesPredictor = getModel(df_hash, K)


# In[12]:


# Use the generated model to make predictions
# These are sample examples
# The idea here is that this model can be used as a component within a bigger project
# Thus we would like the model to behave in a predictable way
# If it doesn't, then the tester should be informed about the change in behavior
print("Example Input 1:", [0,80,40,35,0,43.1,0.58,20])
print("Output 1:", diabetesPredictor[0].predict([[0,80,40,35,0,43.1,0.58,20]]))


# In[13]:

print("Example Input 2:", [0,80,40,35,0,43.1,0.58,20])
print("Output 2:", diabetesPredictor[0].predict([[1,100,50,20,100,43.1,0.688,30]]))


# In[14]:

print("Example Input 3:", [0,80,40,35,0,43.1,0.58,20])
print("Output 3:", diabetesPredictor[0].predict([[2,120,60,20,160,33.1,0.288,40]]))


# In[15]:

print("Example Input 4:", [0,80,40,35,0,43.1,0.58,20])
print("Output 4:", diabetesPredictor[0].predict([[0,140,90,15,168,26.1,2.888,50]]))


# In[16]:

print("Example Input 5:", [0,80,40,35,0,43.1,0.58,20])
print("Output 5:", diabetesPredictor[0].predict([[4,160,40,0,90,23.1,1.288,60]]))


# In[17]:

print("Example Input 6:", [0,80,40,35,0,43.1,0.58,20])
print("Output 6:", diabetesPredictor[0].predict([[5,180,50,35,80,28.1,3.288,50]]))