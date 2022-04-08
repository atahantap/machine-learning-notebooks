#!/usr/bin/env python
# coding: utf-8

# # Homework 02
# 
# ### Atahan Tap
# ### 69374

# ## Importing Data

# In[67]:


import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd

data_set_labels = np.genfromtxt("hw02_data_set_labels.csv", converters = {0: lambda n : int(n)})
data_set_images = np.genfromtxt("hw02_data_set_images.csv", delimiter = ",")

training_set = []
test_set = []

for i in range(len(data_set_images)):
    if i%39 < 25:
        training_set.append(data_set_images[i])
    else:
        test_set.append(data_set_images[i])
        
training_set = np.array(training_set)
test_set = np.array(test_set)

training_set_labels = []
test_set_labels = []

for i in range(len(data_set_labels)):
    if i%39 < 25:
        training_set_labels.append(data_set_labels[i])
    else:
        test_set_labels.append(data_set_labels[i])
        
training_set_labels = np.array(training_set_labels)
test_set_labels = np.array(test_set_labels)

print('Data set divided into training set and test set with corresponding labels. \n')


# ## Calculating Parameters

# In[68]:


K = np.max(data_set_labels)
PIXELS = 20 * 16

pcd = np.array([[np.mean(training_set[training_set_labels == (c + 1)][:,i]) 
                           for i in range(PIXELS)] 
                          for c in range(K)])

print('Probabilities on pixels: \n', pcd)


# In[69]:


class_priors = np.array([np.mean(data_set_labels == c + 1) for c in range(K)])

print('Class Priors: \n', class_priors)


# ## Score functions

# In[70]:


def safelog(n):
    return np.log(n+1e-100)


# In[71]:


def score_classifier(image):
    scores = [(score(image, c), c + 1)
             for c in range(K)]
    return max(scores, key = lambda i : i[0])[1]

def score(image, c):
    sum_score = 0
    for i in range(len(image)):
        sum_score += safelog(pcd[c][i]) if image[i] == 1 else safelog(1 - pcd[c][i])
    sum_score += class_priors[c]
    return sum_score
        


# ## Predictions for training set

# In[72]:


training_set_pred = [score_classifier(image) for image in training_set]


# In[73]:


confusion_matrix_training = pd.crosstab(training_set_pred,
                               training_set_labels,
                               rownames = ['y_pred'],
                               colnames = ['y_truth'])
print('Confusion matrix for training set:\n')
print(confusion_matrix_training)


# ## Predictions for test set

# In[74]:


test_set_pred = [score_classifier(image) for image in test_set]


# In[76]:


confusion_matrix_test = pd.crosstab(test_set_pred,
                               test_set_labels,
                               rownames = ['y_pred'],
                               colnames = ['y_truth'])
print('Confusion matrix for test set:\n')
print(confusion_matrix_test)


# In[ ]:




