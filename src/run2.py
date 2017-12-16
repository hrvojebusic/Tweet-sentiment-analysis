
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

# In[2]:


# Folder paths
DATA_PATH = os.path.join('..', 'data')
OUTPUT_PATH = os.path.join('..', 'output')

# Training data set paths
POS_TRAIN_PATH = os.path.join(DATA_PATH, 'dataset', 'pos_train.txt')
NEG_TRAIN_PATH = os.path.join(DATA_PATH, 'dataset', 'neg_train.txt')
POS_TRAIN_FULL_PATH = os.path.join(DATA_PATH, 'dataset', 'train_pos_full.txt')
NEG_TRAIN_FULL_PATH = os.path.join(DATA_PATH, 'dataset', 'train_neg_full.txt')

# Testing data set paths
TEST_PATH = os.path.join(DATA_PATH, 'dataset', 'test_data.txt')

# Sentiment corpus
POS_WORDS_PATH = os.path.join(DATA_PATH, 'sentiment', 'positive-words.txt')
NEG_WORDS_PATH = os.path.join(DATA_PATH, 'sentiment', 'negative-words.txt')

with open(POS_TRAIN_PATH, 'r') as f:
    pos_data = f.read().splitlines()
with open(NEG_TRAIN_PATH, 'r') as f:
    neg_data = f.read().splitlines()
with open(POS_TRAIN_FULL_PATH, 'r') as f:
    pos_full_data = f.read().splitlines()
with open(NEG_TRAIN_FULL_PATH, 'r') as f:
    neg_full_data = f.read().splitlines()
    
with open(TEST_PATH, 'r') as f:
    test_data = f.read().splitlines()

with open(POS_WORDS_PATH, 'r') as f:
    pos_words = f.read().splitlines()
with open(NEG_WORDS_PATH, 'r') as f:
    neg_words = f.read().splitlines()

stopwords_eng = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()


# ### Tweet preprocessing

# In[3]:


def remove_stopwords(words):
    return list(filter(lambda x: x not in stopwords_eng, words))

def is_number(word):
    original_word = word
    special_chars = ['.', ',', '/', '%', '-']
    for char in special_chars:
        word = word.replace(char, '')
    if word.isdigit():
        return ''
    else:
        return original_word
    
def remove_numbers(words):
    return list(map(is_number, words))

def replace_hashtags(words):
    new_words = []
    
    for word in words:
        if word and word[0]!='#':
            new_words.append(word)
            continue
        
        for hash_word in infer_spaces(word[1:]).split(' '):
            new_words.append(hash_word)
    
    return new_words

# Credit: https://stackoverflow.com/questions/10072744/remove-repeating-characters-from-words
def emphasize(word):
    if (word[:2]=='..'):
        return '...'
    
    new_word = re.sub(r'(.)\1{2,}', r'\1', word)
    if len(new_word) != len(word):
        return '<<' + spell(new_word) + '>>'
    else:
        return word

def emphasize_words(words):
    return list(map(emphasize, words))

def lemmatize(word):
    try:
        temp = lemmatizer.lemmatize(word).lower()
        return temp
    except:
        return word

def normalize_words(words):
    return list(map(lemmatize, words))

def infer_sentiment(words):
    new_words = []
    
    for word in words:
        if (word[:2]=='<<'):
            check_word = word[2:-2]
        else:
            check_word = word
        
        if check_word in pos_words:
            new_words += ['<<positive>>', word]
        elif check_word in neg_words:
            new_words += ['<<negative>>', word]
        else:
            new_words.append(word)
    
    return new_words

def emphasize_punctuation(words):
    special_chars = ['!', '?']
    i = 1
    
    while (i<len(words)):
        word1 = words[i-1]
        word2 = words[i]
        if (word1 in special_chars and word2 in special_chars):
            start = i-1
            while (i+1<len(words) and words[i+1] in special_chars):
                i += 1
            words = words[:start] + ['<<emphasis>>'] + words[i+1:]
            i = start
        
        i += 1
    
    return words

def replace_emoticons(tweet):
    emoticons =     [
     ('<<positive>>',[ ':-)', ':)', '(:', '(-:', \
                       ':-D', ':D', 'X-D', 'XD', 'xD', \
                       '<3', ':\*', ';-)', ';)', ';-D', ';D', '(;', '(-;', ] ),\
     ('<<negative>>', [':-(', ':(', '(:', '(-:', ':,(',\
                       ':\'(', ':"(', ':((', ] ),\
    ]

    def replace_parenth(arr):
        return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]
    
    def regex_join(arr):
        return '(' + '|'.join( arr ) + ')'

    emoticons_regex = [ (repl, re.compile(regex_join(replace_parenth(regx))) )             for (repl, regx) in emoticons ]
    
    for (repl, regx) in emoticons_regex :
        tweet = re.sub(regx, ' '+repl+' ', tweet)
    
    return tweet

def lower(tweet):
    return tweet.lower()


# In[4]:


def parse_tweet(t):
    t = lower(t)
    t = replace_emoticons(t)
    words = t.split(' ')
    words = remove_stopwords(words)
    words = remove_numbers(words)
    words = replace_hashtags(words)
    words = emphasize_words(words)
    words = normalize_words(words)
    words = infer_sentiment(words)
    words = emphasize_punctuation(words)
    tweet = ' '.join(words)
    return tweet

def parse_data(data):
    parsed = []
    start_time = time.time()
    length = len(data)
    
    for i, t in enumerate(data):
        if (i+1)%100000==0:
            print(str(i+1)+'/'+str(len(data)), time.time()-start_time)
        parsed.append(parse_tweet(t))
    
    print('Total time (s): ' + str(time.time()-start_time))
    return parsed


# In[ ]:


print('Positives: ' + str(len(pos_full_data)))
print('Negatives: ' + str(len(neg_full_data)))

'''
start = time.time()
parsed_pos = parse_data(pos_full_data)
parsed_neg = parse_data(neg_full_data)
end = time.time()
print(end - start)
'''


# In[5]:


PARSED_TWEETS_PATH = os.path.join(OUTPUT_PATH, 'parsed_tweets.csv')


# In[6]:

'''
pos_df = pd.DataFrame(parsed_pos)
pos_df.columns = ['text']
pos_df['label'] = 'pos'

neg_df = pd.DataFrame(parsed_neg)
neg_df.columns = ['text']
neg_df['label'] = 'neg'

df = pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

df.to_csv(PARSED_TWEETS_PATH, index=False)
'''

# In[7]:


df = pd.read_csv(PARSED_TWEETS_PATH)


# In[8]:


tweets_flat = df.as_matrix(['text']).astype(str).flatten()
tweets_sentiment_flat = df.as_matrix(['label']).flatten()


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(tweets_flat, tweets_sentiment_flat, test_size=0.20, random_state=42)


# In[10]:


import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below


# In[11]:


n_dim = 200
tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab([x.split(' ') for x in X_train])


# In[12]:


tweet_w2v.train([x.split(' ') for x in X_train], total_examples=len(X_train), epochs=5)


# In[13]:


tweet_w2v.most_similar('positive')     


# In[14]:


vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.split(' ') for x in X_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))


# In[15]:


def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


# In[16]:


from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in map(lambda x: x.split(' '), X_train)])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in map(lambda x: x.split(' '), X_test)])
test_vecs_w2v = scale(test_vecs_w2v)


# In[17]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Conv2D, AveragePooling1D, MaxPooling1D, Dropout, Flatten, Embedding


# In[18]:


temp = pd.DataFrame(y_train)
temp[0] = temp[0].apply(lambda x: 1 if x == 'pos' else -1)
temp_train = temp.as_matrix()


# In[19]:


temp = pd.DataFrame(y_test)
temp[0] = temp[0].apply(lambda x: 1 if x == 'pos' else -1)
temp_test = temp.as_matrix()


# In[24]:


temp = np.expand_dims(train_vecs_w2v, axis=2)


# In[26]:


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

#model.fit(temp, temp_train, epochs=9, batch_size=32, verbose=2)


# ## Classification

# In[ ]:
'''

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

'''



