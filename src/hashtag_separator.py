#https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words/11642687#11642687
import os
import numpy as np
import re

from math import log

# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
DATA_PATH = os.path.join('..','data','meta','words-by-frequency.txt')
words = open(DATA_PATH).read().split()
wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))



def split_hashtag(hashtag):
    try: return infer_spaces(hashtag[1:]).strip()
    except: return hashtag[1:]

    

def get_word_vectors(tweets, glove):
    """
    Creates word embeddings
    INPUT:
        tweets: object containing tweets
        glove: glove dictionary
    OUTPUT:
        embeddings: matrix containing embedings for each word in every tweet
    """
    embeddings = np.zeros((tweets.shape[0], 40, 200, 1))
    for i, tweet in enumerate(tweets['tweet']):
        words = re.split(r'\s+', tweet)
        word_counter = 0
        embeddings_counter = 0
        
        for k in range(40):
            if k<len(words):
                word = words[word_counter]
                try:
                    embeddings[i, embeddings_counter, :, :] = glove[word].reshape((1,1,-1,1))
                    word_counter+=1
                    embeddings_counter+=1
                except:
                    if (not word.startswith("#")):
                        word = "#" + word
                    tokens=split_hashtag(word)
                    for token in tokens.split():
                        if((len(token) != 1) or (token == "a") or (token == "i")):
                            try:
                                embeddings[i, embeddings_counter, :, :] = words[token].reshape((1,1,-1,1))
                                embeddings_counter += 1
                            except:
                                continue
                    word_counter += 1
                    continue
    return embeddings