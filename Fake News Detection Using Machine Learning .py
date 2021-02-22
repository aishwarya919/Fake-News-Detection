#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[2]:


df_fake = pd.read_csv('C:/Users/User/Downloads/Fake News Detection/Fake.csv')
df_true = pd.read_csv('C:/Users/User/Downloads/Fake News Detection/True.csv')


# In[3]:


df_fake.head(10)


# In[4]:


df_true.head(10)


# In[5]:


df_fake["class"] = 0
df_true["class"] = 1


# In[6]:


df_fake.shape, df_true.shape


# ### Merging of Datasets (True & False)

# In[7]:


df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)


# In[8]:


df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv('C:/Users/User/Downloads/Fake News Detection/manual_testing.csv')


# In[9]:


df_marge = pd.concat([df_fake, df_true], axis =0 )
df_marge.head(10)


# In[10]:


df = df_marge.drop(["title", "subject","date"], axis = 1)
df.head(10)


# In[11]:


df = df.sample(frac = 1)


# In[12]:


df.head(10)


# In[13]:


df.isnull().sum()


# ### Cleaning the Dataset

# In[14]:


def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[15]:


df["text"] = df["text"].apply(word_drop)


# In[16]:


df.head(10)


# In[17]:


x = df["text"]
y = df["class"]


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[20]:


vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# ### Logistic Regression

# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


LR = LogisticRegression()
LR.fit(xv_train,y_train)


# In[23]:


LR.score(xv_test, y_test)


# In[24]:


pred_lr=LR.predict(xv_test)


# In[25]:


print(classification_report(y_test, pred_lr))


# ### Decision Tree Classification

# In[26]:


from sklearn.tree import DecisionTreeClassifier


# In[27]:


DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[28]:


DT.score(xv_test, y_test)


# In[29]:


pred_dt = DT.predict(xv_test)


# In[30]:


print(classification_report(y_test, pred_dt))


# ### Random Forest Classification

# In[31]:


from sklearn.ensemble import RandomForestClassifier


# In[32]:


RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)


# In[33]:


RFC.score(xv_test, y_test)


# In[34]:


pred_rfc = RFC.predict(xv_test)


# In[35]:


print(classification_report(y_test, pred_rfc))


# ### Manual Testing  

# In[36]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(word_drop) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]), 
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_RFC[0])))


# In[37]:


news = str(input())
manual_testing(news)


# In[ ]:




