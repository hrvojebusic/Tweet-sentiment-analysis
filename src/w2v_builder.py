import numpy as np
import re


def create_w2v(tweets, text_column, size, w2v):
    content = tweets[text_column].values
    content = list(map(lambda x: re.split(r'\s+', x), content))
    content = list(map(lambda x: build_word_vector(x, size, w2v), content))
    return np.concatenate(content)


def build_word_vector(words, size, w2v):
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
