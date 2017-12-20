import numpy as np
import re


def create_w2v(tweets, text_column, size, w2v):
    """
    Creates word embeddings vector representation for the
    tweets passed as data frame, in specified column. Vector
    size is specified by the second argumnent and needs to
    correspond to the vector size of the given pretrained
    vocabulary - the third argument.
    INPUT:
        tweets: data frame containing tweets
        text_column: name of the column where tweet content is stored
        size: vector size
        w2v: pretrained vocabulary vector mapping
    OUTPUT:
        word embeddings vectors for the given tweets
    """
    content = tweets[text_column].values
    content = list(map(lambda x: re.split(r'\s+', x), content))
    content = list(map(lambda x: build_word_vector(x, size, w2v), content))
    return np.concatenate(content)


def build_word_vector(words, size, w2v):
    """
    Creates word embeddings vector representation for the
    words passed as the first argument. Vector size is specified
    by the second arguments and needs to correspond to the
    vector size of the given pretrained vocabulary - the third
    argument.
    INPUT:
        words: words whose vector representation is queried
        size: vector size
        w2v: pretrained vocabulary vector mapping
    OUTPUT:
        word embeddings vector for the given words
    """
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in words:
        try:
            vec += w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec
