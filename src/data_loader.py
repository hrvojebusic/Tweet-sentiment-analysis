import pandas as pd

from os import path

def load_train_data():
    pos_path = path.join('..', 'data', 'parsed', 'train_pos_full.csv')
    neg_path = path.join('..', 'data', 'parsed', 'train_neg_full.csv')

    pos_data = pd.read_csv(pos_path, header=None)
    pos_data.columns = ['text']
    pos_data['sentiment'] = 1

    neg_data = pd.read_csv(neg_path, header=None)
    neg_data.columns = ['text']
    neg_data['sentiment'] = -1

    train_data = pd.concat([pos_data, neg_data], axis=0)
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    return train_data

def load_test_data():
    test_path = path.join('..', 'data', 'parsed', 'test_full.csv')
    
    test_data = pd.read_csv(test_path, header=None)
    test_data.columns = ['id', 'text']
    test_data = test_data.set_index('id')
    test_data.text = test_data.text.astype(str)
    
    return test_data

def save_submission(results):
    submission_path = path.join('..', 'output', 'submission.csv')
    results.to_csv(submission_path)
