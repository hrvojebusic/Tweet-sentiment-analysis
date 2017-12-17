
# coding: utf-8

# ### Libraries

# In[1]:


import csv
import nltk
import os
import pandas as pd
import re
import time
import numpy as np

from autocorrect import spell
from hashtag_separator import infer_spaces
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# ### Data import

# In[152]:


# Training data paths
POS_TRAIN_PATH = os.path.join('..', 'data', 'parsed', 'train_pos_full.csv')
NEG_TRAIN_PATH = os.path.join('..', 'data', 'parsed', 'train_neg_full.csv')

# Test data paths
TEST_PATH = os.path.join('..', 'data', 'parsed', 'test_full.csv')

pos_train_data = pd.read_csv(POS_TRAIN_PATH, header=None)
pos_train_data.columns = ['text']
pos_train_data['sentiment'] = 1

neg_train_data = pd.read_csv(NEG_TRAIN_PATH, header=None)
neg_train_data.columns = ['text']
neg_train_data['sentiment'] = 0

test_data = pd.read_csv(TEST_PATH, header=None)
test_data.columns = ['text']

train_data = pd.concat([pos_train_data, neg_train_data], axis=0)


# In[153]:


import gensim

from gensim.models.word2vec import Word2Vec # the word2vec model gensim class

LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below


# In[154]:


vocab = train_data.text.tolist() + test_data.text.tolist()

n_dim = 200
tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab([x.split(' ') for x in vocab])


# In[156]:


tweet_w2v.train([x.split(' ') for x in vocab], total_examples=len(vocab), epochs=5)


# In[158]:
# tweet_w2v.most_similar('dun')     


# In[124]:
#vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
#matrix = vectorizer.fit_transform([x.split(' ') for x in train_data.text.tolist()])
#tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
#print('vocab size :', len(tfidf))


# In[159]:


def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) #* tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


# In[ ]:


from sklearn.preprocessing import scale

train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in map(lambda x: x.split(' '), train_data.text.tolist())])
train_vecs_w2v = scale(train_vecs_w2v)


# In[ ]:


test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in map(lambda x: x.split(' '), test_data.text.tolist())])
test_vecs_w2v = scale(test_vecs_w2v)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Conv2D, AveragePooling1D, MaxPooling1D, Dropout, Flatten, Embedding


# In[ ]:


temp_train = train_data.sentiment.as_matrix()


# In[ ]:


temp = np.expand_dims(train_vecs_w2v, axis=2)


# In[143]:


model = Sequential()

model.add(Conv1D(128, 2, padding='same', input_shape=(200, 1)))
model.add(Flatten())
model.add(Dropout(0.2))

#model.add(Conv1D(32, 2, border_mode='same'))
#model.add(Conv1D(16, 2, border_mode='same'))
#model.add(Dense(180,activation='sigmoid'))
#model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(temp, temp_train, epochs=1, batch_size=32, verbose=2)


# In[142]:
"""
model = Sequential()

model.add(Conv1D(64, 2, border_mode='same', input_shape=(200, 1)))
model.add(Conv1D(32, 2, border_mode='same'))
model.add(Conv1D(16, 2, border_mode='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(temp, temp_train, epochs=1, batch_size=32, verbose=2)
"""

# In[145]:


temp = np.expand_dims(test_vecs_w2v, axis=2)
y_pred = model.predict(temp)


# In[150]:


pred = pd.DataFrame(y_pred)
pred['Id'] = range(1, len(pred)+1)
pred[0] = pred[0].apply(lambda x: 1 if x >= 0.5 else -1)
pred = pred.set_index('Id')
pred.columns = ['Prediction']


# In[ ]:


path = os.path.join('..', 'output', 'submission.csv')
pred.to_csv(path)


# In[147]:


#model.evaluate(np.expand_dims(test_vecs_w2v, axis=2), temp_test, batch_size=128, verbose=2)


"""
# ## Classification

# In[ ]:


y_tweets = list(map(lambda x: 1 if x == 'pos' else -1, tweets_sentiment_flat))


# In[ ]:


from sklearn.neural_network import MLPClassifier

nn_pipe = Pipeline([
    ('vec', TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True)), 
    ('nn', MLPClassifier(solver='lbfgs', alpha=1e-5,
                         hidden_layer_sizes=(64,), random_state=1))
])


# In[ ]:


len(tweets_flat)


# In[ ]:


start = time.time()
nn_pipe.fit(tweets_flat, y_tweets)
end = time.time()
print(end - start)


# In[ ]:


from sklearn.svm import LinearSVC

pipe = Pipeline([
    ('vec', TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True)), 
    ('svm', LinearSVC())
])

param_grid = {
    'svm__C' : [0.1, 1, 5, 10],
}

CV_pipe = GridSearchCV(pipe, param_grid=param_grid, cv=2)


# In[ ]:


from sklearn.naive_bayes import *

pipe_nb = Pipeline([
    ('vec', TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True)), 
    ('bayes', MultinomialNB())
])

CV_pipe_nb = GridSearchCV(pipe_nb, param_grid={}, cv=5)


# In[ ]:


start = time.time()
CV_pipe_nb.fit(tweets_flat, y_tweets)
end = time.time()
print(end - start)


# In[ ]:


start = time.time()
CV_pipe.fit(tweets_flat, y_tweets)
end = time.time()
print(end - start)


# In[ ]:


y_pred = nn_pipe.predict(tweets_flat)
print(classification_report(y_tweets, y_pred))  


# In[ ]:


cl_data = []

for i, l in enumerate(test_data):
    l = l.split(',', 1)
    id_ = l[0]
    tweet = l[1]
    cl_data.append([id_, tweet])

df = pd.DataFrame(cl_data)
df.columns = ['Id', 'Tweet']
df.head()


# In[ ]:


df.Tweet = parse_data(df.Tweet)
df.Tweet.head()


# In[ ]:


y_pred = nn_pipe.predict(df['Tweet'].as_matrix().flatten())


# In[ ]:


OUTPUT_FILE_PATH = os.path.join(OUTPUT_PATH, 'submission.csv')

res_df = pd.DataFrame({ 'Id': df['Id'].as_matrix().flatten(),
                        'Prediction': y_pred})

res_df = res_df.set_index('Id')
res_df.to_csv(OUTPUT_FILE_PATH)


# #### Neural net

# In[ ]:


from sklearn.neural_network import MLPClassifier
"""

