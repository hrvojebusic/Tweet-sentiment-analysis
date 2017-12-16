import numpy as np
import os
import pandas as pd
import re
import string
import time

from autocorrect import spell
from hashtag_separator import infer_spaces
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

def remove_user(tweet):
    """
    Removes '<user>' tag from a tweet.
    INPUT: 
        tweet: original tweet as a string
    OUTPUT: 
        tweet with <user> tags removed
    """
    return tweet.replace('<user>', '')

def remove_url(tweet):
    """
    Removes '<url>' tag from a tweet.
    INPUT: 
        tweet: original tweet as a string
    OUTPUT: 
        tweet with <url> tags removed
    """
    return tweet.replace('<url>', '')

def remove_numbers(tweet):
    """
    Removes numbers from a tweet.
    INPUT:
        tweet: original tweet as a string
    OUTPUT:
        tweet with all numbers removed
    """
    special_chars = ['.', ',', ':', '+', '-', '*', '/', '%', '_']
    words = re.split(r'\s+', tweet)
    new_words = []
    
    for word in words:    
        for char in special_chars:
            word = word.replace(char, '')
        if word.isdigit():
            continue
        else:
            new_words.append(word)
        
    return ' '.join(new_words)

def replace_emoticons(tweet):
    """
    Replaces emoticons in tweet with descriptive tags.
    INPUT:
        tweet: original tweet as a string
    OUTPU:
        tweet with emoticons replaced
    """
    heart_emoticons = ('<<positive>>', ['<3', '❤']) 

    positive_emoticons = ('<<positive>>', [':)', ';)', ':]', ':-]', ':-)', ';-)', ":')", ':*', ':-*', 
                                              ':D', ':-D', '8-D', 'xD', 'XD', ':P', ':-P',
                                              '(:', '(;', '[:', '[-:', '(-:', '(-;', "(':", '*:', '*-:', ])
    
    negative_emoticons = ('<<negative>>', [':(', ':((', ':-(', ":'(",
                                              '):', ')):', ')-:', ")':"])
    
    emoticons = [heart_emoticons, negative_emoticons]

    def replace_parenth(arr):
        return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]
    
    def regex_join(arr):
        return '(' + '|'.join( arr ) + ')'

    emoticons_regex = [ (repl, re.compile(regex_join(replace_parenth(regx))) ) \
            for (repl, regx) in emoticons ]
    
    for (repl, regx) in emoticons_regex :
        tweet = re.sub(regx, ' '+repl+' ', tweet)
    
    return tweet

def split_hashtags(tweet):
    """
    Tweet whose hashtags will be split into words that
    are mentioned in those hashtags.
    INPUT:
        tweet: original tweet as a string
    OUTPUT:
        tweet with hashtags replaced with words
        mentioned in those hashtags
    """
    words = re.split(r'\s+', tweet)
    new_words = []
    
    for word in words:
        if word and word[0]!='#':
            new_words.append(word)
            continue

        for hash_word in infer_spaces(word[1:].lower()).split(' '):
            new_words.append(hash_word)
    
    return ' '.join(new_words)

def emphasize_punctuation(tweet):
    words = re.split(r'\s+', tweet)
    new_words = []
    translation = str.maketrans("","", string.punctuation)

    for word in words:
        nubbed = word.replace('?', '').replace('!', '').replace('.', '')

        if not nubbed and len(word) >= 2:
            new_words.append('<<emphasis>>')
        
        if len(word) != len(nubbed):
            new_words.append(word.translate(translation))
        else:
            new_words.append(word)
    
    return ' '.join(new_words)

def lower(tweets):
    """
    Lowers all characters in tweets.
    INPUT:
        tweets: list of tweets
    OUTPUT:
        list of tweets with all characters lowered
    """
    return [tweet.lower() for tweet in tweets]

def remove_stopwords(tweet, list_of_stopwords=None):
    """
    Removes stopwords from the tweet. Arbitrary stopword list 
    can be specified through the second method argument. List 
    of english stopwords will be used if no other stopwords 
    are specified.
    INPUT:
        tweet: original tweet as a string
        list_of_stopwords: list of stopwords
    OUTPUT:
        words filtered from stopwords
    """
    words = re.split(r'\s+', tweet)
    if list_of_stopwords == None:
        list_of_stopwords = stopwords.words('english')
    words = list(filter(lambda x: x not in list_of_stopwords, words))
    return ' '.join(words)

def infer_sentiment(tweet, positive_words, negative_words):
    """

    INPUT:
        tweet: original tweet
    OUTPUT:
        tweet 
    """
    words = re.split(r'\s+', tweet)
    new_words = []
    
    for word in words:
        if word in positive_words:
            new_words += ['<<positive>>', word]
        elif word in negative_words:
            new_words += ['<<negative>>', word]
        else:
            new_words.append(word)
    
    return ' '.join(new_words)

def stem_single(word, stemmer):
    """
    Stemms single word.
    INPUT:
        word: word to be stemmed
        stemmer: stemmer to be used for stemming
    OUTPUT:
        stemmed word
    """
    return stemmer.stem(word)

def stem(tweet, stemmer=None):
    """
    Stemms all words in a tweet with the given stemmer. 
    If no stemmer is provided, default 'LancasterStemmer'
    is used.
    INPUT:
        tweet: tweet with words to be stemmed
        stemmer: stemmer to be used for stemming
    OUPUT:
        original tweet with all its words stemmed
    """
    words = re.split(r'\s+', tweet)
    if stemmer == None:
        stemmer = LancasterStemmer()
    words = list(map(lambda x: stem_single(x, stemmer), words))
    return ' '.join(words)

def lemmatize_single(word, lemmatizer):
    """
    Lemmatizes single word. If the word can not be lemmatized, it
    will be returned in its original form.
    INPUT:
        word: word to be lemmatized
        lemmatizer: lemmatizer to be used for lemmatization
    OUTPUT:
        lemmatized word or original word if lemmatization could
        not be performed
    """
    try:
        temp = lemmatizer.lemmatize(word).lower()
        return temp
    except:
        return word

def lemmatize(tweet, lemmatizer=None):
    """
    Lemmatizes all words in a tweet with the given lemmatizer.
    If no lemmatizer is provided, default 'WordNetLemmatizer'
    is used.
    INPUT:
        tweet: tweet with words to be lemmatized
        lemmatizer: lemmatizer to be used for lemmatization
    OUTPUT:
        original tweet with all its words lemmatized
    """
    words = re.split(r'\s+', tweet)
    if lemmatizer == None:
        lemmatizer = WordNetLemmatizer()
    words = list(map(lambda x: lemmatize_single(x, lemmatizer), words))
    return ' '.join(words)

def preprocess_tweets(tweets, parameters=None):
    if parameters == None: # Set default parameters
        parameters = {
            'remove_user_tags' : True,
            'remove_url_tags' : True,
            'replace_emoticons' : True,
            'split_hashtags' : True,
            'emphasize_punctuation': True,
            'remove_numbers' : True,
            'remove_stopwords' : True,
            'infer_sentiment' : True,
            'stem' : True,
            'lemmatize' : True
        }

    start_time = time.time()

    if parameters['remove_user_tags']:
        tweets = list(map(remove_user, tweets))
        print('Removing USER tags: FINISHED')

    if parameters['remove_url_tags']:
        tweets = list(map(remove_url, tweets))
        print('Removing URL tags: FINISHED')

    if parameters['replace_emoticons']:
        tweets = list(map(replace_emoticons, tweets))
        print('Replacing emoticons: FINISHED')

    if parameters['split_hashtags']:
        tweets = list(map(split_hashtags, tweets))
        print('Splitting hashtags: FINISHED')

    if parameters['emphasize_punctuation']:
        tweets = list(map(emphasize_punctuation, tweets))
        print('Emphasizing punctuation: FINISHED')

    if parameters['remove_numbers']:
        tweets = list(map(remove_numbers, tweets))
        print('Removing numbers: FINISHED')

    if parameters['remove_stopwords']:
        list_of_stopwords = stopwords.words('english')
        tweets = list(map(lambda x: remove_stopwords(x, list_of_stopwords), tweets))
        print('Removing stopwords: FINISHED')

    if parameters['infer_sentiment']:
        pw = load_positive_words()
        nw = load_negative_words()
        tweets = list(map(lambda x: infer_sentiment(x, pw, nw), tweets))
        print('Inferring sentiment: FINISHED')

    if parameters['stem']:
        stemmer = LancasterStemmer()
        tweets = list(map(lambda x: stem(x, stemmer), tweets))
        print('Stemming: FINISHED')

    if parameters['lemmatize']:
        lemmatizer = WordNetLemmatizer()
        tweets = list(map(lambda x: lemmatize(x, lemmatizer), tweets))
        print('Lemmatizing: FINISHED')

    end_time = time.time()
    print('Time elapsed (s): {}'.format(end_time - start_time))

    return tweets

def load_positive_words():
    path = os.path.join('..', 'data', 'sentiment', 'positive-words.txt')
    with open(path, 'r') as f:
        pos_words = f.read().splitlines()
    return pos_words

def load_negative_words():
    path = os.path.join('..', 'data', 'sentiment', 'negative-words.txt')
    with open(path, 'r') as f:
        neg_words = f.read().splitlines()
    return neg_words

print(preprocess_tweets(['12%323 Sebastijan ! <user> <url> touched !! me in inappropriate places, he is a douche :( #helloworld'], None))