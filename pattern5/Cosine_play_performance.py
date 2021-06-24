#!/usr/bin/env python
# coding: utf-8

# In[4]:


# get_ipython().system('pip -V')


# In[1]:


# get_ipython().system('pip list')


# In[1]:


# get_ipython().system('pip install pythainlp==2.3.1 --user')


# In[1]:


import deepcut
import pandas as pd
import numpy as np
import re
from itertools import chain
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from flask import Flask, jsonify, request
import threading 
import time
import json
import requests
import joblib
import pickle


# In[3]:


# pd.show_versions()


# ## data from all category

# In[2]:


def tokenize_text_list(ls):
    """Tokenize list of text"""
    return list(chain.from_iterable([deepcut.tokenize(ls)]))


# In[3]:


exit_flag = True
beforeTok={}

def getQuestions():
    global beforeTok
    while True:
#         raw_questions = backendAPI()
        raw_questions = getQuestionsFromBackendAPI() # <-- Uncomment this to get real questions
        
        # do tokenize
        tokenize_questions = raw_questions
        
        curriculumDF = pd.DataFrame(data=tokenize_questions['หลักสูตร'])
        curriculumDF = curriculumDF.rename(columns={0:"curriculum"})
        
        admissionDF = pd.DataFrame(data=tokenize_questions['การรับเข้านักศึกษา'])
        admissionDF = admissionDF.rename(columns={0:"admission"})
        
        enrollmentDF = pd.DataFrame(data=tokenize_questions['ลงทะเบียนเรียน'])
        enrollmentDF = enrollmentDF.rename(columns={0:"enrollment"})
        
        faqDF = pd.DataFrame(data=tokenize_questions['คำถามทั่วไป'])
        faqDF = faqDF.rename(columns={0:"faq"})
        
        tokenized_enrollment =enrollmentDF.enrollment.map(tokenize_text_list)
        tokenized_admission =admissionDF.admission.map(tokenize_text_list)
        tokenized_curriculum =curriculumDF.curriculum.map(tokenize_text_list)
        tokenized_faq =faqDF.faq.map(tokenize_text_list)
        
#         beforeTok = {}
        beforeTok['ลงทะเบียนเรียน'] = enrollmentDF
        beforeTok['การรับเข้านักศึกษา'] = admissionDF
        beforeTok['หลักสูตร'] = curriculumDF
        beforeTok['คำถามทั่วไป'] = faqDF
        
        # Update question
        questions_data = tokenize_questions
        if exit_flag: 
            break
        
        # Set query time
        time.sleep(259200)


# In[4]:


url = 'https://natthawat.live/api'


# In[5]:


def getQuestionsFromBackendAPI():
    response = requests.get('%s/km/faq' % url)
    faqs = json.loads(response.text)

    response = requests.get('%s/km/category' % url)
    categories = json.loads(response.text)

    questions_data = {}
    for category in categories:
        questions_data[category['category']] = []
        for faq in faqs:
            if faq['category']['category'] == category['category']:
                questions_data[category['category']].append(faq['question'])
                
    return questions_data


# In[6]:


getQuestionsThread = threading.Thread(target = getQuestions)


# In[7]:


getQuestionsThread.start()


# In[8]:


get_ipython().run_cell_magic('time', '', 'getQuestions()')


# ## category

# In[9]:


data = pd.read_excel("Category.xlsx")


# In[10]:


#Load File
with open('token_text_category.data', 'rb') as filehandle:
    # read the data as binary data stream
    tokenized_texts = pickle.load(filehandle)


# In[11]:


def text_to_bow(tokenized_text, vocabulary_):
    n_doc = len(tokenized_text)
    values, row_indices, col_indices = [], [], []
    for r, tokens in enumerate(tokenized_text):
        feature = {}
        for token in tokens:
            word_index = vocabulary_.get(token)
            if word_index is not None:
                if word_index not in feature.keys():
                    feature[word_index] = 1
                else:
                    feature[word_index] += 1
        for c, v in feature.items():
            values.append(v)
            row_indices.append(r)
            col_indices.append(c)
        #print(feature)

    # document-term matrix in sparse CSR format
    X = sp.csr_matrix((values, (row_indices, col_indices)),
                      shape=(n_doc, len(vocabulary_)))
    return X

vocabulary_ = {v: k for k, v in enumerate(set(chain.from_iterable(tokenized_texts)))}
X = text_to_bow(tokenized_texts, vocabulary_)


# In[12]:


transformer = TfidfTransformer()
svd_model = TruncatedSVD(n_components=100,
                         algorithm='arpack', n_iter=100)
X_tfidf = transformer.fit_transform(X)
X_svd = svd_model.fit_transform(X_tfidf)


# In[13]:


tag = pd.get_dummies(data.Category).columns


# In[14]:


#Load Model
logist_models = joblib.load("category_model.pkl")


# In[15]:


y_pred = np.argmax(np.vstack([model.predict_proba(X_svd)[:, 1] for model in logist_models]).T, axis=1)
y_pred = np.array([tag[yi] for yi in y_pred])
y_true = data.Category.values
print(tag[0:4])


# In[16]:


#Clean Text
def remove_repettition(text):
    token_list = list(text)
    if len(token_list) > 2:
        filter_list = [True, True]
        n = len(token_list)
        for i in range(2, n):
            if (token_list[i] == token_list[i-1]) and (token_list[i] == token_list[i-2]):
                filter_list.append(False)
            else:
                filter_list.append(True)

        output = ''.join(np.array(token_list)[filter_list])
    else:
        output = text
    return output

def cleansing(text):
    # \t, \n, \xa0 and other special characters. Replace by blank string
    text = re.sub('[\t\n\xa0\"\'!?\/\(\)%\:\=\-\+\*\_ๆ]', '', text)
    
    # Numbers. Replace by space
    text = re.sub('[0-9]', ' ', text)
    
    # Dot. Replace by space
    text = re.sub('[\.]', ' ', text)
    
    # One or more consecutive space. Replace by single space
    text = re.sub('\s+',' ',text)
    
    # Remove 2 or more repettition
    text = remove_repettition(text)
    
    return text


# In[17]:


textInput = 'วิศวคอมพิวเตอร์มีหลักสูตรอะไรบ้าง'
textInput= cleansing(textInput)
tokenized_text = deepcut.tokenize(textInput)
x = text_to_bow([tokenized_text], vocabulary_)
x_tfidf = transformer.transform(x)
x_svd = svd_model.transform(x_tfidf)
pred = [model.predict_proba(x_svd.reshape(-1, 1).T).ravel()[1] for model in logist_models]
print(list(zip(tag, pred)))
predict_category = max(list(zip(tag, pred)))


# In[18]:


max_value = 0
max_category = ''
pred_results = list(zip(tag, pred))
for pred_result in pred_results:
  # print(pred_result)
  if pred_result[1] > max_value:
    max_value = pred_result[1]
    max_category = pred_result[0]
print(max_category, max_value)


# ## prediction

# In[19]:


#cosine
from pythainlp import word_tokenize 
from pythainlp.word_vector import * 
from sklearn.metrics.pairwise import cosine_similarity  
model=get_model()
def sentence_vectorizer(ss,dim=300,use_mean=True): 
    s = word_tokenize(ss)
    vec = np.zeros((1,dim))
    for word in s:
        if word in model.wv.index2word:
            vec+= model.wv.word_vec(word)
        else: pass
    if use_mean: vec /= len(s)
    return vec
def sentence_similarity(s1,s2):
    return cosine_similarity(sentence_vectorizer(str(s1)),sentence_vectorizer(str(s2)))


# In[21]:


len(beforeTok[max_category])


# In[22]:


word = beforeTok[max_category].iloc[0]


# In[23]:


word[0]


# In[24]:


word


# In[25]:


import json


# In[26]:


result = []
for index in range(len(beforeTok[max_category])):
#     print(beforeTok[max_category].iloc[index])
    word = beforeTok[max_category].iloc[index]
#     print(word[0])
#     word = beforeTok[max_category].iloc[index]
    subResult = sentence_similarity(textInput,word[0])
    subResult = subResult[0][0]
    result.append(subResult)


# In[27]:


max_question_percentage = max(result)
print(max_question_percentage)
question_index = np.where(result == max_question_percentage)
print(question_index)
predictedQuestion = beforeTok[max_category].loc[question_index[0][0]]
predictedQuestion = predictedQuestion[0]


# In[28]:


predictedQuestion


# # time total

# In[29]:


get_ipython().run_cell_magic('time', '', 'textInput = "วิศวคอมมีหลักสูตรอะไรบ้าง"\ntextInput= cleansing(textInput)\ntokenized_text = deepcut.tokenize(textInput)\nx = text_to_bow([tokenized_text], vocabulary_)\nx_tfidf = transformer.transform(x)\nx_svd = svd_model.transform(x_tfidf)\npred = [model.predict_proba(x_svd.reshape(-1, 1).T).ravel()[1] for model in logist_models]\nprint(list(zip(tag, pred)))\npredict_category = max(list(zip(tag, pred)))\n\nmax_value = 0\nmax_category = \'\'\npred_results = list(zip(tag, pred))\nfor pred_result in pred_results:\n  # print(pred_result)\n    if pred_result[1] > max_value:\n        max_value = pred_result[1]\n        max_category = pred_result[0]\nprint(max_category, max_value)\n\nresult = []\nfor index in range(len(beforeTok[max_category])):\n    word = beforeTok[max_category].iloc[index]\n    subResult = sentence_similarity(textInput,word[0])\n    subResult = subResult[0][0]\n    result.append(subResult)\nmax_question_percentage = max(result)\nprint(max_question_percentage)\nquestion_index = np.where(result == max_question_percentage)\nprint(question_index)\n#         print(beforeTok[max_category].loc[question_index[0][0]])\npredictedQuestion = beforeTok[max_category].loc[question_index[0][0]]\npredictedQuestion = predictedQuestion[0]\nprint(predictedQuestion)\nvalue = {\n    "category": max_category,\n    "accuracy": "%lf" % max_value,\n    "predictedQuestion": str(predictedQuestion),\n    "similarity": "%lf" % max_question_percentage\n}')


# # time seperate

# In[30]:


get_ipython().run_cell_magic('time', '', 'textInput = "วิศวคอมมีหลักสูตรอะไรบ้าง"\ntextInput= cleansing(textInput)')


# In[31]:


get_ipython().run_cell_magic('time', '', 'tokenized_text = deepcut.tokenize(textInput)')


# In[32]:


get_ipython().run_cell_magic('time', '', 'x = text_to_bow([tokenized_text], vocabulary_)\nx_tfidf = transformer.transform(x)\nx_svd = svd_model.transform(x_tfidf)')


# In[33]:


get_ipython().run_cell_magic('time', '', 'pred = [model.predict_proba(x_svd.reshape(-1, 1).T).ravel()[1] for model in logist_models]\n# print(list(zip(tag, pred)))')


# In[34]:


get_ipython().run_cell_magic('time', '', "predict_category = max(list(zip(tag, pred)))\nmax_value = 0\nmax_category = ''\npred_results = list(zip(tag, pred))")


# In[35]:


get_ipython().run_cell_magic('time', '', 'for pred_result in pred_results:\n  # print(pred_result)\n    if pred_result[1] > max_value:\n        max_value = pred_result[1]\n        max_category = pred_result[0]\nprint(max_category, max_value)\n# end of category model')


# In[36]:


get_ipython().run_cell_magic('time', '', 'result = []\nfor index in range(len(beforeTok[max_category])):\n    word = beforeTok[max_category].iloc[index]\n    subResult = sentence_similarity(textInput,word[0])\n    subResult = subResult[0][0]\n    result.append(subResult)')


# In[37]:


get_ipython().run_cell_magic('time', '', 'max_question_percentage = max(result)\n# print(max_question_percentage)\nquestion_index = np.where(result == max_question_percentage)\n# print(question_index)\n# print(beforeTok[max_category].loc[question_index[0][0]])\npredictedQuestion = beforeTok[max_category].loc[question_index[0][0]]\npredictedQuestion = predictedQuestion[0]\n# print(predictedQuestion)')


# In[38]:


get_ipython().run_cell_magic('time', '', 'value = {\n    "category": max_category,\n    "accuracy": "%lf" % max_value,\n    "predictedQuestion": str(predictedQuestion),\n    "similarity": "%lf" % max_question_percentage\n}')

