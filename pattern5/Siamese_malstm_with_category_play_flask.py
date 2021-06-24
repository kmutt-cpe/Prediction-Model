#!/usr/bin/env python
# coding: utf-8

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


# In[8]:


get_ipython().system('pip install nltk --user')


# In[10]:


get_ipython().system('pip install xlrd')


# In[12]:


get_ipython().system('pip install openpyxl')


# In[2]:


#prediction
from pythainlp.corpus import thai_stopwords
import deepcut
from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

import itertools
import datetime
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint
import difflib


# In[3]:


# Question import
import requests

# Category model import
import pickle
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import joblib


# In[4]:


# from flask_ngrok import run_with_ngrok
from flask import Flask, jsonify, request

import threading 
import json
import time
import requests


# ## Category_model

# In[11]:


data = pd.read_excel("Category.xlsx")
data


# In[6]:


#Load File
with open('token_text_category.data', 'rb') as filehandle:
    # read the data as binary data stream
    tokenized_texts = pickle.load(filehandle)


# In[7]:


from itertools import chain
def tokenize_text_list(ls):
    """Tokenize list of text"""
    return list(chain.from_iterable([deepcut.tokenize(ls)]))


# In[13]:


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


# In[14]:


transformer = TfidfTransformer()
svd_model = TruncatedSVD(n_components=100,
                         algorithm='arpack', n_iter=100)
X_tfidf = transformer.fit_transform(X)
X_svd = svd_model.fit_transform(X_tfidf)


# In[15]:


tag = pd.get_dummies(data.Category).columns


# In[16]:


#Lib
import joblib

#Load Model
logist_models = joblib.load("category_model.pkl")


# In[17]:


y_pred = np.argmax(np.vstack([model.predict_proba(X_svd)[:, 1] for model in logist_models]).T, axis=1)
y_pred = np.array([tag[yi] for yi in y_pred])
y_true = data.Category.values
print(tag[0:4])


# ## Prediction

# In[18]:


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


# In[19]:


import gensim
wv_model = gensim.models.Word2Vec.load('corpus.th.model')


# In[20]:


def word2idx(word):
    index = 0
    index = wv_model.wv.vocab[word].index
    return index


# In[21]:


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


# In[22]:


# define word embedding
vocab_list = [(k, wv_model.wv[k]) for k, v in wv_model.wv.vocab.items()]
embeddings_matrix = np.zeros((len(wv_model.wv.vocab.items()) + 1, wv_model.vector_size))
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    embeddings_matrix[i + 1] = vocab_list[i][1]


# In[23]:


# vocab_list


# In[24]:


EMBEDDING_DIM = 300
embeddings_matrix = 1 * np.random.randn(len(vocab_list) + 1, EMBEDDING_DIM)  # This will be the embedding matrix
embeddings_matrix[0] = 0  # So that the padding will be ignored


# In[25]:


# Model variables
n_hidden = 256
batch_size = 128
n_epoch = 100
max_seq_length = 2704


# In[26]:


# embeddings_matrix


# In[27]:


def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


# In[28]:


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


# In[29]:


malstm.summary()


# In[30]:


# Load best weight from model
malstm.load_weights('sm_colab_ka.h5')


# #Test with Text

# In[31]:


def prepare_for_predict(input_questions):
    q_input= []
    cleansing(input_questions)
    tokenized_input_1 =deepcut.tokenize(input_questions)
    for sentence in tokenized_input_1:
      q_input.append(sentence)
    q_input= word_index(tokenized_input_1)
    q_input = pad_sequences(q_input, maxlen=max_seq_length)
    return q_input


# In[32]:


max_word = 19219
max_seq_length = 2704


# In[33]:


#Duplicate list
def duplicate(testList, n):
    return [ele for ele in testList for _ in range(n)]


# ## data from all category

# In[34]:


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


# In[35]:


url = 'https://natthawat.live/api'


# In[36]:


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


# In[37]:


getQuestionsThread = threading.Thread(target = getQuestions)


# In[38]:


getQuestionsThread.start()


# In[39]:


print(beforeTok)


# # Run API

# In[42]:


app = Flask(__name__)
@app.route('/prediction', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        data = request.get_json()
        inputQuestion = data["inputQuestion"]
        #category model
        inputQuestion= cleansing(inputQuestion)
        tokenized_input_2 = inputQuestion
        tokenized_text = deepcut.tokenize(tokenized_input_2)
        x = text_to_bow([tokenized_text], vocabulary_)
        x_tfidf = transformer.transform(x)
        x_svd = svd_model.transform(x_tfidf)
        pred = [model.predict_proba(x_svd.reshape(-1, 1).T).ravel()[1] for model in logist_models]

        # print(list(zip(tag, pred)))
        predict_category = max(list(zip(tag, pred)))
        max_value = 0
        max_category = ''
        pred_results = list(zip(tag, pred))

        for pred_result in pred_results:
            # print(pred_result)
            if pred_result[1] > max_value:
                max_value = pred_result[1]
                max_category = pred_result[0]
        # print(max_category, max_value)
        # end of category model


        # print(tokenized[max_category])


        # prediction
        tokenized_category = tokenized[max_category]
        # print(tokenized_category)
        max_word = 19219
        max_seq_length = 2704
        q_category= []
        for sentence in tokenized_category:
            q_category.append(sentence)
        q_category = word_index(q_category)
        all_Question_categorylen = len(q_category)
        # all_Question_categorylen

        tokenized_dup_input_2= duplicate([tokenized_input_2],all_Question_categorylen)
        # print(tokenized_dup_input_2)

        q_user = word_index(tokenized_dup_input_2)
        # Split to dicts
        M_input = {'left': q_category, 'right': q_user}
        # Zero padding
        for model_input, side in itertools.product([M_input], ['left', 'right']):
            model_input[side] = pad_sequences(model_input[side], maxlen=max_seq_length)

        # Make sure everything is ok
        assert M_input['left'].shape == M_input['right'].shape
        play_predict = malstm.predict(x=[M_input['left'], M_input['right']])

        max_question_percentage = max(play_predict)
        # print(max_question_percentage)
        question_index = np.where(play_predict == max_question_percentage)
        # print(question_index)
        predictedQuestion = beforeTok[max_category].loc[question_index[0][0]]
        predictedQuestion = predictedQuestion[0]

        value = {
            "category": max_category,
            "accuracy": "%lf" % max_value,
            "predictedQuestion": str(predictedQuestion),
            "similarity": "%lf" % max_question_percentage
        }
        return json.dumps(value, ensure_ascii=False).encode('utf8')
    else:
        return "Hello. I am alive!"

app.run(port=3000,debug=False)


# # example of api

# In[41]:


{
"inputQuestion":"วิศวคอมมีหลักสูตรอะไรบ้าง"
}

