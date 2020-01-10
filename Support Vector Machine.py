#!/usr/bin/env python
# coding: utf-8

# # NLP Project Sentiment Analysis

# ## Support Vector Machine

# In[1]:


#importing lib
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read data
corpus = pd.read_csv('E:/Semester 7/NLP/Tugas Akhir/IMDB Dataset.csv')


# In[3]:


print('Panjang Corpus : ',len(corpus))


# In[4]:


corpus.head()


# In[5]:


corpus.tail()


# In[6]:


#cek review pos dan neg dalam corpus
review_positif = len([x for x in corpus['sentiment'] if x == 'positive'])
review_negatif = len([x for x in corpus['sentiment'] if x == 'negative'])
print('Review Positif : ',review_positif)
print('Review Negatif : ',review_negatif)


# In[7]:


import string
#remove punctuation
no_punctuations=[]
i=0
for word in corpus['review']:
    for punctuation in string.punctuation:
        word = word.replace(punctuation,"")
    for number in '1234567890':
        word = word.replace(number,"")
    corpus['review'][i] = word
    i = i+1
    
corpus['review'].head()


# In[8]:


x = corpus['review']
y = corpus['sentiment']


# In[9]:


#stopword remove, dan tokenisasi
vect = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 4)


# In[10]:


#create X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=1, test_size= 0.2)


# In[11]:


#Using training data to transform text into counts of features for each review
vect.fit(X_train)


# In[12]:


X_train_dtm = vect.transform(X_train) 


# In[13]:


X_test_dtm = vect.transform(X_test)


# In[14]:


#Creating model using multinomial naive bayes
NB = MultinomialNB()
NB.fit(X_train_dtm, y_train)
y_pred = NB.predict(X_test_dtm)


# In[15]:


#Check the pos vs neg review
tokens_words = vect.get_feature_names()
print('Analysis')
print('No. of tokens: ',len(tokens_words))
counts = NB.feature_count_
df_table = {'Token':tokens_words,'Negative': counts[0,:],'Positive': counts[1,:]}
tokens = pd.DataFrame(df_table, columns= ['Token','Positive','Negative'])
positives = len(tokens[tokens['Positive']>tokens['Negative']])
negatives = len(tokens_words)-positives
print('No. of positive tokens: ',positives)
print('No. of negative tokens: ',negatives)


# In[16]:


#Accuracy SVM
SVM = LinearSVC()
SVM.fit(X_train_dtm, y_train)
y_pred = SVM.predict(X_test_dtm)
print('Support Vector Machine')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')

