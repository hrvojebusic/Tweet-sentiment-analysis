import numpy as np
import pandas as pd

from os import path


def load_train_data(pos_file_name='train_pos_full.csv', neg_file_name='train_neg_full.csv'):
    """
    Loads both positive and negative training tweets from specified files.
    Files need to be placed in 'data/dataset' directory. If no file names 
    are provided, default full data sets of both positive and negative 
    sentiments are loaded.
    INPUT:
        pos_file_name: name of the file containing tweets of positive sentiment
        neg_file_name: name of the file containing tweets of negative sentiment
    OUTPUT:
        data frame containing training tweets
    """
    pos_path = path.join('..', 'data', 'parsed', pos_file_name)
    neg_path = path.join('..', 'data', 'parsed', neg_file_name)

    pos_data = pd.read_csv(pos_path, header=None)
    pos_data.columns = ['text']
    pos_data['sentiment'] = 1

    neg_data = pd.read_csv(neg_path, header=None)
    neg_data.columns = ['text']
    neg_data['sentiment'] = -1

    train_data = pd.concat([pos_data, neg_data], axis=0)
    return train_data


def load_glove_data():
    """
    Loads Stanford's dictionary of word embeddings created by using corpus of
    Twitter posts. Word embeddings are vectors of 200 components.
    OUTPUT:
        dictionary containing tweet word embeddings
    """
    glove_path = path.join('..', 'data', 'glove', 'glove.twitter.27B.200d.txt')
    f = open(glove_path,'r')
    
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    
    return model


def load_test_data(test_file_name='test_full.csv'):
    """
    Loads test tweets from a specified file. File needs to be placed 
    in 'data/dataset' directory. If no file name is provided, default 
    test data set is loaded.
    INPUT:
        test_file_name: name of the file containing test tweets
    OUTPUT:
        data frame containing test tweets
    """
    test_path = path.join('..', 'data', 'parsed', test_file_name)
    
    test_data = pd.read_csv(test_path, header=None)
    test_data.columns = ['id', 'text']
    test_data = test_data.set_index('id')
    test_data.text = test_data.text.astype(str)
    
    return test_data


def save_submission(results, file_name='submission.csv'):
    """
    Persists data frame containing submissions for online evaluation
    in the 'output' directory. If no specific file name is provided,
    submission file will be called 'submission.csv'.
    INPUT:
        results: data frame containing predictions for test data
        file_name: name to be used for the submission file
    """
    submission_path = path.join('..', 'output', file_name)
    results.to_csv(submission_path)
