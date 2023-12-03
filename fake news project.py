#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline 
from sklearn.linear_model import PassiveAggressiveClassifier 
from sklearn.naive_bayes import MultinomialNB
import pickle


# #  Loading the dataset

# In[22]:


dataset = pd.read_csv('fake_or_real_news.csv')


# # Extract features (text) and labels from the dataset

# In[23]:


x = dataset['text']
y = dataset['label']


#  # Display the first few rows and shape of the dataset

# In[24]:


dataset.shape


# In[25]:


dataset.head()


# # Check for missing values in the dataset

# In[26]:


dataset.isnull().any()


# # Split the dataset into training and testing sets
# 

# In[27]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# # Initialize TfidfVectorizer

# In[28]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)


# # Transform the training and testing text data

# In[29]:


tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)


# #  Train the Passive Aggressive Classifier

# In[30]:


pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
y_pred = pac.predict(tfidf_test)


# #  Evaluate the Passive Aggressive Classifier

# In[31]:


score = accuracy_score(y_test, y_pred)
print(f'Passive Aggressive Classifier Accuracy: {round(score * 100, 2)}%')


# # Create a pipeline with TfidfVectorizer and Multinomial Naive Bayes

# In[32]:


pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('nbmodel', MultinomialNB())])



#  # Train the pipeline
# 

# In[33]:


pipeline.fit(x_train, y_train)


# # Evaluate the pipeline

# In[34]:


score = pipeline.score(x_test, y_test)
print(f'Multinomial Naive Bayes Accuracy: {round(score * 100, 2)}')


# #  Predict using the pipeline
# 

# In[35]:


pred = pipeline.predict(x_test)


# #  Display classification report and confusion matrix for Multinomial Naive Bayes

# In[36]:


print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))


# #  Save the Multinomial Naive Bayes model to a file

# In[37]:


with open('model.pkl', 'wb') as handle:
    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




