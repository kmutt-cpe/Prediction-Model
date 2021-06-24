#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip list')


# In[1]:


get_ipython().system('pip install numpy==1.19.5 --user')


# In[2]:


get_ipython().system('pip install h5py==2.10.0 --user')


# In[3]:


get_ipython().system('pip install gensim==3.6.0 --user')


# In[4]:


get_ipython().system('pip install deepcut --user')


# In[5]:


get_ipython().system('pip install pythainlp --user')


# In[6]:


get_ipython().system('pip install nltk --user')


# In[12]:


#prediction
import deepcut
import pandas as pd
import numpy as np
import re
from itertools import chain
import itertools
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Lambda
import tensorflow.keras.backend as K


# In[13]:


# Question import
import requests

# Category model import
import pickle
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import joblib


# In[14]:


# from flask_ngrok import run_with_ngrok
from flask import Flask
import threading 
import json
import time


# ## Category_model

# In[15]:


data = pd.read_excel("Category.xlsx")
data


# In[16]:


#Load File
with open('token_text_category.data', 'rb') as filehandle:
    # read the data as binary data stream
    tokenized_texts = pickle.load(filehandle)


# In[17]:


def tokenize_text_list(ls):
    """Tokenize list of text"""
    return list(chain.from_iterable([deepcut.tokenize(ls)]))


# In[18]:


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


# In[19]:


transformer = TfidfTransformer()
svd_model = TruncatedSVD(n_components=100,
                         algorithm='arpack', n_iter=100)
X_tfidf = transformer.fit_transform(X)
X_svd = svd_model.fit_transform(X_tfidf)


# In[20]:


tag = pd.get_dummies(data.Category).columns


# In[21]:


#Load Model
logist_models = joblib.load("category_model.pkl")


# In[22]:


y_pred = np.argmax(np.vstack([model.predict_proba(X_svd)[:, 1] for model in logist_models]).T, axis=1)
y_pred = np.array([tag[yi] for yi in y_pred])
y_true = data.Category.values
print(tag[0:4])


# ## Prediction

# In[23]:


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


# In[24]:


import gensim
wv_model = gensim.models.Word2Vec.load('corpus.th.model')


# In[25]:


def word2idx(word):
    index = 0
    index = wv_model.wv.vocab[word].index
    return index


# In[26]:


def word_index(listword):
    dataset = []
    vocabulary = dict()
    inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
    for sentence in listword:
        tmp = []
        for w in sentence:
            if w not in wv_model:
                continue

            if w not in vocabulary:
                vocabulary[w] = len(inverse_vocabulary)
                tmp.append(len(inverse_vocabulary))
                inverse_vocabulary.append(w)
            else:
                tmp.append(word2idx(w))
        dataset.append(tmp)
    return np.array(dataset)


# In[27]:


# define word embedding
vocab_list = [(k, wv_model.wv[k]) for k, v in wv_model.wv.vocab.items()]
embeddings_matrix = np.zeros((len(wv_model.wv.vocab.items()) + 1, wv_model.vector_size))
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    embeddings_matrix[i + 1] = vocab_list[i][1]


# In[28]:


# vocab_list


# In[29]:


EMBEDDING_DIM = 300
embeddings_matrix = 1 * np.random.randn(len(vocab_list) + 1, EMBEDDING_DIM)  # This will be the embedding matrix
embeddings_matrix[0] = 0  # So that the padding will be ignored


# In[30]:


# Model variables
n_hidden = 256
batch_size = 128
n_epoch = 100
max_seq_length = 2704


# In[31]:


# embeddings_matrix


# In[32]:


def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


# In[33]:


# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

embedding_layer = Embedding(len(embeddings_matrix), EMBEDDING_DIM, weights=[embeddings_matrix], input_length=max_seq_length, trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])


malstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Start training
training_start_time = time.time()


# In[34]:


malstm.summary()


# In[35]:


# Load best weight from model
malstm.load_weights('sm_colab_ka.h5')


# #Test with Text

# In[36]:


def prepare_for_predict(input_questions):
    q_input= []
    cleansing(input_questions)
    tokenized_input_1 =deepcut.tokenize(input_questions)
    for sentence in tokenized_input_1:
      q_input.append(sentence)
    q_input= word_index(tokenized_input_1)
    q_input = pad_sequences(q_input, maxlen=max_seq_length)
    return q_input


# In[37]:


max_word = 19219
max_seq_length = 2704


# In[38]:


#Duplicate list
def duplicate(testList, n):
    return [ele for ele in testList for _ in range(n)]


# ## data from all category

# In[39]:


exit_flag = False
beforeTok={}
tokenized = {}

def getQuestions():
    global beforeTok
    global tokenized
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
        
        tokenized = {}
        tokenized['ลงทะเบียนเรียน'] = tokenized_enrollment
        tokenized['การรับเข้านักศึกษา'] = tokenized_admission
        tokenized['หลักสูตร'] = tokenized_curriculum
        tokenized['คำถามทั่วไป'] = tokenized_faq
        
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


# In[40]:


url = 'https://natthawat.live/api'


# In[41]:


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


# In[42]:


getQuestionsThread = threading.Thread(target = getQuestions)


# In[43]:


getQuestionsThread.start()


# In[46]:


print(beforeTok)


# # time total

# In[47]:


get_ipython().run_cell_magic('time', '', 'inputQuestion = "วิศวคอมมีหลักสูตรอะไรบ้าง"\n# print(\'input: \' + inputQuestion)\n\n#category model\ninputQuestion= cleansing(inputQuestion)\ntokenized_input_2 = inputQuestion\ntokenized_text = deepcut.tokenize(tokenized_input_2)\nx = text_to_bow([tokenized_text], vocabulary_)\nx_tfidf = transformer.transform(x)\nx_svd = svd_model.transform(x_tfidf)\npred = [model.predict_proba(x_svd.reshape(-1, 1).T).ravel()[1] for model in logist_models]\n\n# print(list(zip(tag, pred)))\npredict_category = max(list(zip(tag, pred)))\nmax_value = 0\nmax_category = \'\'\npred_results = list(zip(tag, pred))\n\nfor pred_result in pred_results:\n    # print(pred_result)\n    if pred_result[1] > max_value:\n        max_value = pred_result[1]\n        max_category = pred_result[0]\n# print(max_category, max_value)\n# end of category model\n\n\n# print(tokenized[max_category])\n\n\n# prediction\ntokenized_category = tokenized[max_category]\n# print(tokenized_category)\nmax_word = 19219\nmax_seq_length = 2704\nq_category= []\nfor sentence in tokenized_category:\n    q_category.append(sentence)\nq_category = word_index(q_category)\nall_Question_categorylen = len(q_category)\n# all_Question_categorylen\n\ntokenized_dup_input_2= duplicate([tokenized_input_2],all_Question_categorylen)\n# print(tokenized_dup_input_2)\n\nq_user = word_index(tokenized_dup_input_2)\n# Split to dicts\nM_input = {\'left\': q_category, \'right\': q_user}\n# Zero padding\nfor model_input, side in itertools.product([M_input], [\'left\', \'right\']):\n    model_input[side] = pad_sequences(model_input[side], maxlen=max_seq_length)\n\n# Make sure everything is ok\nassert M_input[\'left\'].shape == M_input[\'right\'].shape\nplay_predict = malstm.predict(x=[M_input[\'left\'], M_input[\'right\']])\n\nmax_question_percentage = max(play_predict)\n# print(max_question_percentage)\nquestion_index = np.where(play_predict == max_question_percentage)\n# print(question_index)\npredictedQuestion = beforeTok[max_category].loc[question_index[0][0]]\npredictedQuestion = predictedQuestion[0]\n\nvalue = {\n    "category": max_category,\n    "accuracy": "%lf" % max_value,\n    "predictedQuestion": str(predictedQuestion),\n    "similarity": "%lf" % max_question_percentage\n}')


# # time seperate

# In[47]:


get_ipython().run_cell_magic('time', '', 'inputQuestion = "วิศวคอมมีหลักสูตรอะไรบ้าง"\nprint(\'input: \' + inputQuestion)')


# In[48]:


get_ipython().run_cell_magic('time', '', '#category model\ntokenized_input_2= cleansing(inputQuestion)')


# In[49]:


get_ipython().run_cell_magic('time', '', 'tokenized_text = deepcut.tokenize(tokenized_input_2)')


# In[50]:


get_ipython().run_cell_magic('time', '', 'x = text_to_bow([tokenized_text], vocabulary_)\nx_tfidf = transformer.transform(x)\nx_svd = svd_model.transform(x_tfidf)')


# In[51]:


get_ipython().run_cell_magic('time', '', 'pred = [model.predict_proba(x_svd.reshape(-1, 1).T).ravel()[1] for model in logist_models]\n# print(list(zip(tag, pred)))')


# In[52]:


get_ipython().run_cell_magic('time', '', "predict_category = max(list(zip(tag, pred)))\nmax_value = 0\nmax_category = ''\npred_results = list(zip(tag, pred))")


# In[53]:


get_ipython().run_cell_magic('time', '', 'for pred_result in pred_results:\n    # print(pred_result)\n    if pred_result[1] > max_value:\n        max_value = pred_result[1]\n        max_category = pred_result[0]\nprint(max_category, max_value)\n# end of category model')


# In[54]:


get_ipython().run_cell_magic('time', '', '# prediction\ntokenized_category = tokenized[max_category]\n# print(tokenized_category)\nmax_word = 19219\nmax_seq_length = 2704')


# In[55]:


get_ipython().run_cell_magic('time', '', 'q_category= []\nfor sentence in tokenized_category:\n    q_category.append(sentence)\nq_category = word_index(q_category)\nall_Question_categorylen = len(q_category)\n# all_Question_categorylen')


# In[56]:


get_ipython().run_cell_magic('time', '', 'tokenized_input_2 = deepcut.tokenize(inputQuestion)')


# In[57]:


get_ipython().run_cell_magic('time', '', 'tokenized_dup_input_2= duplicate([tokenized_input_2],all_Question_categorylen)')


# In[58]:


get_ipython().run_cell_magic('time', '', "q_user = word_index(tokenized_dup_input_2)\n# Split to dicts\nM_input = {'left': q_category, 'right': q_user}\n# Zero padding\nfor model_input, side in itertools.product([M_input], ['left', 'right']):\n    model_input[side] = pad_sequences(model_input[side], maxlen=max_seq_length)")


# In[59]:


get_ipython().run_cell_magic('time', '', "# Make sure everything is ok\nassert M_input['left'].shape == M_input['right'].shape")


# In[60]:


get_ipython().run_cell_magic('time', '', "play_predict = malstm.predict(x=[M_input['left'],  M_input['right']])")


# In[61]:


get_ipython().run_cell_magic('time', '', 'max_question_percentage = max(play_predict)\nprint(max_question_percentage)\nquestion_index = np.where(play_predict == max_question_percentage)\nprint(question_index)\npredictedQuestion = beforeTok[max_category].loc[question_index[0][0]]\npredictedQuestion = predictedQuestion[0]')


# In[62]:


get_ipython().run_cell_magic('time', '', 'value = {\n    "category": max_category,\n    "accuracy": "%lf" % max_value,\n    "predictedQuestion": str(predictedQuestion),\n    "similarity": "%lf" % max_question_percentage\n}')


# In[ ]:




