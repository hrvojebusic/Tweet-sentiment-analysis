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
from unicodedata import numeric


def check_tweet_emphasis(tweet):
    """
    Performs check whether words in a tweet have 3 or 
    more repeating characters. If the word has 3 or 
    more repeating characters, word will be replaced
    by one that matches the original word as close as 
    possible repeated twice, otherwise original word 
    is returned.
    INPUT:
        tweet: original tweet as a string
    OUTPUT:
        tweet with appropriate words emphasized
    """
    def check_word_emphasis(word):
        new_word = re.sub(r'(.)\1{2,}', r'\1', word)
        if len(new_word) != len(word):
            spelled_word = spell(new_word)
            return spelled_word + ' ' + spelled_word
        else:
            return word

    words = re.split(r'\s+', tweet)
    words = list(map(check_word_emphasis, words)) 
    return ' '.join(words)


def expand_contractions(tweet):
    """
    Expands language contractions found in the English vocabulary
    in the tweet.
    INPUT:
        tweet: original tweet as a string
    OUTPUT:
        tweet with its contractions expanded
    """
    tweet = re.sub("can't", 'can not', tweet, flags=re.I)
    tweet = re.sub("n't", ' not', tweet, flags=re.I)
    tweet = re.sub("i'm", 'i am', tweet, flags=re.I)
    tweet = re.sub("'re", ' are', tweet, flags=re.I)
    tweet = re.sub("it's", 'it is', tweet, flags=re.I)    
    tweet = re.sub("that's", 'that is', tweet, flags=re.I)
    tweet = re.sub("'ll", ' will', tweet, flags=re.I)
    tweet = re.sub("'l", ' will', tweet, flags=re.I)
    tweet = re.sub("'ve", ' have', tweet, flags=re.I)
    tweet = re.sub("'d", ' would', tweet, flags=re.I)
    tweet = re.sub("he's", 'he is', tweet, flags=re.I)
    tweet = re.sub("she's", 'she is', tweet, flags=re.I)
    tweet = re.sub("what's", 'what is', tweet, flags=re.I)
    tweet = re.sub("who's", 'who is', tweet, flags=re.I)
    tweet = re.sub("'s", '', tweet, flags=re.I)
    return tweet


def infer_sentiment(tweet, positive_words, negative_words):
    """
    Expands tweet with positive or negative emphasis depending on
    whether word from a tweet is a part of collection of positive 
    or negative words.
    INPUT:
        tweet: original tweet as a string
    OUTPUT:
        tweet with positive and negative words emphasized
    """
    words = re.split(r'\s+', tweet)
    new_words = []
    
    for word in words:
        if word in positive_words:
            new_words.append('positive')
        elif word in negative_words:
            new_words.append('negative')
        new_words.append(word)
    
    return ' '.join(new_words)


def lemmatize(tweet, lemmatizer=None):
    """
    Lemmatizes all words in a tweet with the given lemmatizer.
    If no lemmatizer is provided, default 'WordNetLemmatizer'
    is used.
    INPUT:
        tweet: original tweet as a string
        lemmatizer: lemmatizer to be used for lemmatization
    OUTPUT:
        tweet with all of its words lemmatized
    """
    def lemmatize_single(word, lemmatizer):
        try:
            return lemmatizer.lemmatize(word).lower()
        except:
            return word

    words = re.split(r'\s+', tweet)
    if lemmatizer == None:
        lemmatizer = WordNetLemmatizer()
    words = list(map(lambda x: lemmatize_single(x, lemmatizer), words))
    return ' '.join(words)


def remove_non_characters(tweet):
    """
    Removes non-characters from a tweet.
    INPUT:
        tweet: original tweet as a string
    OUTPUT:
        tweet with non-characters removed
    """
    tweet = re.sub("[\"\'\.\,\:\;\@\_\-\+\*\\\/\%\_\(\)\[\]\{\}\&\!\?\~\=\^]", '', tweet)
    return ' '.join([w if w!='<' and w!='>' else '' for w in re.split(r'\s+', tweet)])


def remove_numbers(tweet):
    """
    Replaces all numbers in a tweet with the word
    'number'.
    INPUT:
        tweet: original tweet as a string
    OUTPUT:
        tweet with all numbers removed
    """
    words = re.split(r'\s+', tweet)
    new_words = []

    for word in words:
        if bool(re.search(r'\d', word)):
            new_words.append('number')
        else:
            new_words.append(word)
        
    return ' '.join(new_words)


def remove_small_words(tweet):
    """
    Removes small words from a tweet. Small words are those with
    only one character.
    INPUT:
        tweet: original tweet as a string
    OUTPUT:
        tweet filtered from small words
    """
    return ' '.join([word for word in tweet.split() if not word.isalpha() or len(word) > 1])


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
        tweet filtered from stopwords
    """
    words = re.split(r'\s+', tweet)
    if list_of_stopwords == None:
        list_of_stopwords = stopwords.words('english')
    words = list(filter(lambda x: x not in list_of_stopwords, words))
    return ' '.join(words)


def remove_url(tweet):
    """
    Removes '<url>' tag from a tweet.
    INPUT: 
        tweet: original tweet as a string
    OUTPUT: 
        tweet with <url> tags removed
    """
    return tweet.replace('<url>', '')


def remove_user(tweet):
    """
    Removes '<user>' tag from a tweet.
    INPUT: 
        tweet: original tweet as a string
    OUTPUT: 
        tweet with <user> tags removed
    """
    return tweet.replace('<user>', '')


def replace_emoticons(tweet):
    """
    Replaces emoticons in a tweet with descriptive word.
    INPUT:
        tweet: original tweet as a string
    OUTPU:
        tweet with emoticons replaced
    """
    heart_emoticons = ['<3', '❤']

    positive_emoticons = [
        ':)', ';)', ':-)', ';-)', ":')", ':*', ':-*', '=)', '[:', '[-:',
        ':D', ':-D', '8-D', 'xD', 'XD', ':P', ':-P', ':p', ':-p', ':d', ';d',
        '(:', '(;', '(-:', '(-;', "(':", '*:', '*-:', '(=', ':]', ':-]'
        ]
    
    negative_emoticons = [
        ':(', ':((', ';(', ':-(', ":'(", '=(',
        '):', ')):', ');', ')-:', ")':", ')='
        ]
    
    words = re.split(r'\s+', tweet)
    new_words = []

    for word in words:
        if word in heart_emoticons or word in positive_emoticons:
            new_words.append('positive')
        elif word in negative_emoticons:
            new_words.append('negative')
        else:
            new_words.append(word)
    
    return ' '.join(new_words)


def replace_emoticons_with_tags(tweet):
    """
    Replaces emoticons in a tweet with tags.
    INPUT:
        tweet: original tweet as a string
    OUTPU:
        tweet with emoticons replaced
    """
    hearts = set(['<3', '❤'])

    happy_faces = set([
        ':)', ":')", '=)', ':-)', ':]', ":']", '=]', ':-]', ':d',
        '(:', "(':", '(=', '(-:', '[:', "[':", '[=', '[-:' 
        ])

    sad_faces = set([
        ':(', ":'(", '=(', ':-(', ':[', ":'[", '=[', ':-[',
        '):', ")':", ')=', ')-:', ']:', "]':", ']=', ']-:'
    ])

    neutral_faces = set([
        ':/', ':\\', ':|',
        '/:', '\\:', '|:'
    ])

    cheeky_faces = set([
        ':P', ':p', ":'P", ":'p", '=P', '=p', ':-P', ":-p"
    ])

    words = re.split(r'\s+', tweet)
    new_words = []

    for word in words:
        if word in hearts:
            new_words.append('<heart>')
        elif word in happy_faces:
            new_words.append('<smile>')
        elif word in neutral_faces:
            new_words.append('<neutralface>')
        elif word in sad_faces:
            new_words.append('<sadface>')
        elif word in cheeky_faces:
            new_words.append('<lolface>')
        else:
            new_words.append(word)

    return ' '.join(new_words)


def split_hashtags(tweet):
    """
    Splits all tweet hashtags into words that are
    mentioned in those hashtags.
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
    def stem_single(word, stemmer):
        return stemmer.stem(word)

    words = re.split(r'\s+', tweet)
    if stemmer == None:
        stemmer = LancasterStemmer()
    words = list(map(lambda x: stem_single(x, stemmer), words))
    return ' '.join(words)


def tag_hashtags(tweet):
    """
    Marks hashtags in tweet.
    INPUT:
        tweet: original tweet as a string
    OUTPUT:
        tweet with hashtags marked
    """
    words = re.split(r'\s+', tweet)
    new_words = []

    for word in words:
        if word and word[0]=='#':
            new_words.append('<hashtag>')
        new_words.append(word)

    return ' '.join(new_words)


def tag_numbers(tweet):
    """
    Marks numbers in tweet.
    INPUT:
        tweet: original tweet as a string
    OUTPUT:
        tweet with numbers marked
    """
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    words = re.split(r'\s+', tweet)
    new_words = []

    for word in words:
        trimmed = re.sub('[,\.:%_\-\+\*\/\%\_]', '', word)
        if is_number(trimmed):
            new_words.append('<number>')
        new_words.append(word)

    return ' '.join(new_words)


def tag_repeated_characters(tweet):
    """
    Marks words with repeating characters in tweet.
    Repetition needs to be greater or equal than three.
    INPUT:
        tweet: original tweet as a string
    OUTPUT:
        tweet with words with repeating characters marked
    """
    def tag_repeated(word):
        return re.sub(r'([a-z])\1\1+$', r'\1 <elong>', word)

    words = re.split(r'\s+', tweet)
    words = list(map(tag_repeated, words))
    return ' '.join(words)


def tag_repeated_punctuations(tweet):
    """
    Marks repeated punctuations in tweet.
    INPUT:
        tweet: original tweet as a string
    OUTPUT:
        tweet with repeating punctuations marked
    """
    for symb in ['!', '?', '.', ',']:
        regex = '(\\' + symb + '( *)){2,}'
        tweet = re.sub(regex, ' <repeat> ' + symb, tweet)
    return tweet


def preprocess_tweets(tweets, text_column, train=True, parameters=None):
    """
    Performs tweet preprocessing on the data frame passed
    as the first argument with column containing tweets specified.
    Default preprocessing parameters are:
        filter_duplicates = True
        remove_url_tags = True
        remove_user_tags = True
        replace_emoticons_with_tags = False
        tag_hashtags = False
        tag_numbers = False
        tag_repeated_characters = False
        tag_repeated_punctuations = False
        expand_contractions = True
        check_tweet_emphasis = True
        split_hashtags = True
        remove_stopwords = True
        remove_small_words = True
        remove_numbers = True
        infer_sentiment = True
        replace_emoticons = True
        remove_non_characters = True
        stem = False
        lemmatize = False
    Custom preprocessing parameters can be specified.
    INPUT
        tweets: data frame containing original tweets
        text_column: column name containing tweet content in the data frame
        train: specifies whether tweets being processed belong to the training set
        parameters: custom preprocessing parameters
    OUTPUT:
        data frame containing processed tweets
    """

    def load_positive_words():
        path = os.path.join('..', 'data', 'sentiment', 'positive-words.txt')
        with open(path, 'r') as f:
            pos_words = f.read().splitlines()
        return set(pos_words)

    def load_negative_words():
        path = os.path.join('..', 'data', 'sentiment', 'negative-words.txt')
        with open(path, 'r') as f:
            neg_words = f.read().splitlines()
        return set(neg_words)

    if parameters == None: # Set of default parameters
        parameters = {
            'filter_duplicates' : True,
            'remove_url_tags' : True,
            'remove_user_tags' : True,
            'replace_emoticons_with_tags' : False,
            'tag_hashtags' : False,
            'tag_numbers' : False,
            'tag_repeated_characters' : False,
            'tag_repeated_punctuations' : False,
            'expand_contractions' : True,
            'check_tweet_emphasis': True,
            'split_hashtags' : True,
            'remove_stopwords' : True,
            'remove_small_words': True,
            'remove_numbers' : True,
            'infer_sentiment' : True,
            'replace_emoticons' : True,
            'remove_non_characters' : True,
            'stem' : False,
            'lemmatize' : False
        }

    start_time = time.time()
    content = tweets[text_column].copy()

    if parameters['filter_duplicates'] and train:
        content = content.drop_duplicates()
        print('Filtering duplicates: FINISHED')

    if parameters['remove_url_tags']:
        content = list(map(remove_url, content))
        print('Removing URL tags: FINISHED')

    if parameters['remove_user_tags']:
        content = list(map(remove_user, content))
        print('Removing USER tags: FINISHED')

    if parameters['replace_emoticons_with_tags']:
        content = list(map(replace_emoticons_with_tags, content))
        print('Replacing emoticons with tags: FINISHED')

    if parameters['tag_hashtags']:
        content = list(map(tag_hashtags, content))
        print('Tagging hashtags: FINISHED')

    if parameters['tag_numbers']:
        content = list(map(tag_numbers, content))
        print('Tagging numbers: FINISHED')

    if parameters['tag_repeated_characters']:
        content = list(map(tag_repeated_characters, content))
        print('Tagging repeated characters: FINISHED')

    if parameters['tag_repeated_punctuations']:
        content = list(map(tag_repeated_punctuations, content))
        print('Tagging repeated punctuations: FINISHED')

    if parameters['expand_contractions']:
        content = list(map(expand_contractions, content))
        print('Expanding contractions: FINISHED')

    if parameters['check_tweet_emphasis']:
        content = list(map(check_tweet_emphasis, content))
        print('Checking tweet emphasis: FINISHED')
    
    if parameters['split_hashtags']:
        content = list(map(split_hashtags, content))
        print('Splitting hashtags: FINISHED')

    if parameters['remove_stopwords']:
        list_of_stopwords = stopwords.words('english')
        content = list(map(lambda x: remove_stopwords(x, list_of_stopwords), content))
        print('Removing stopwords: FINISHED')

    if parameters['remove_small_words']:
        content = list(map(remove_small_words, content))
        print('Removing small words: FINISHED')

    if parameters['remove_numbers']:
        content = list(map(remove_numbers, content))
        print('Removing numbers: FINISHED')

    if parameters['infer_sentiment']:
        pw = load_positive_words()
        nw = load_negative_words()
        content = list(map(lambda x: infer_sentiment(x, pw, nw), content))
        print('Inferring sentiment: FINISHED')

    if parameters['replace_emoticons']:
        content = list(map(replace_emoticons, content))
        print('Replacing emoticons: FINISHED')

    if parameters['remove_non_characters']:
        content = list(map(remove_non_characters, content))
        print('Removing non-characters: FINISHED')    

    if parameters['stem']:
        stemmer = LancasterStemmer()
        content = list(map(lambda x: stem(x, stemmer), content))
        print('Stemming: FINISHED')

    if parameters['lemmatize']:
        lemmatizer = WordNetLemmatizer()
        content = list(map(lambda x: lemmatize(x, lemmatizer), content))
        print('Lemmatizing: FINISHED')

    end_time = time.time()
    print('Time elapsed (s): {}'.format(end_time - start_time))

    return pd.DataFrame({ 'parsed' : content })
