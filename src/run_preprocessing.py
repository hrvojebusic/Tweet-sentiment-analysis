import sys
import pandas as pd

from os import path
from tweet_preprocessing import preprocess_tweets


def main():
    if len(sys.argv) != 2 or (sys.argv[1] not in ['train', 'test']):
        print("Invalid command. Expected 'train' or 'test'.")
        return

    parameters = {
            'filter_duplicates' : True,
            'remove_url_tags' : False,
            'remove_user_tags' : False,
            'replace_emoticons_with_tags' : True,
            'tag_hashtags' : True,
            'tag_numbers' : True,
            'tag_repeated_characters' : True,
            'tag_repeated_punctuations' : True,
            'expand_contractions' : True,
            'check_tweet_emphasis': False,
            'split_hashtags' : False,
            'remove_stopwords' : False,
            'remove_small_words': False,
            'remove_numbers' : False,
            'infer_sentiment' : True,
            'replace_emoticons' : False,
            'remove_non_characters' : False,
            'stem' : False,
            'lemmatize' : False,
            'spell_unrecognized': False,
            'remove_like': False
        }

    if sys.argv[1] == 'train':
        process_train_data(
            dest_pos_name='train_pos_full.csv',
            dest_neg_name='train_neg_full.csv',
            parameters=parameters
            )
    if sys.argv[1] == 'test':
        process_test_data(
            dest_name='test_full.csv',
            parameters=parameters
            )
    

def process_train_data(dest_pos_name='train_pos_full.csv', dest_neg_name='train_neg_full.csv', parameters=None):
    """
    Processes train tweets. If custom processing parameters are not 
    defined, default processing parameters will be used as defined
    in 'tweet_preprocessing' module.
    INPUT:
        dest_pos_name: name of the .csv file where processed positive 
        tweets will be saved
        dest_neg_name: name of the .csv file where processed negative
        tweets will be saved
        parameters: custom processing parameters
    """
    path_read_pos_train = path.join('..', 'data', 'dataset', 'train_pos_full.txt')
    pos_df = read_train_data(path_read_pos_train)
    print('Positive training tweets read. Count: {}'.format(len(pos_df)))

    path_read_neg_train = path.join('..', 'data', 'dataset', 'train_neg_full.txt')
    neg_df = read_train_data(path_read_neg_train)
    print('Negative training tweets read. Count: {}'.format(len(neg_df)))

    print('Processing positive tweets...')
    pos_df = preprocess_tweets(pos_df, 'text', parameters=parameters)
    print('Processing negative tweets...')
    neg_df = preprocess_tweets(neg_df, 'text', parameters=parameters)

    path_save_pos_train = path.join('..', 'data', 'parsed', dest_pos_name)
    pos_df.to_csv(path_save_pos_train, header=False, index=False)
    path_save_neg_train = path.join('..', 'data', 'parsed', dest_neg_name)
    neg_df.to_csv(path_save_neg_train, header=False, index=False)
    print('Processed training tweets have been successfully saved.')


def process_test_data(dest_name='test_full.csv', parameters=None):
    """
    Processes test tweets. If custom processing parameters are not 
    defined, default processing parameters will be used as defined
    in 'tweet_preprocessing' module.
    INPUT:
        dest_name: name of the .csv file where processed tweets will be saved
        parameters: custom processing parameters
    """
    path_read_test = path.join('..', 'data', 'dataset', 'test_data.txt')
    test_df = read_test_data(path_read_test)
    print('Test tweets read. Count: {}'.format(len(test_df)))

    print('Processing test tweets...')
    test_df.text = preprocess_tweets(test_df, 'text', train=False, parameters=parameters)

    path_save_test = path.join('..', 'data', 'parsed', dest_name)
    test_df.to_csv(path_save_test, header=False, index=False)
    print('Processed test tweets have been successfully saved.')


def read_train_data(file_path):
    """
    Reads training tweets.
    INPUT:
        file_path: path to file containing training set of tweets
    OUTPUT:
        data frame containing training set of tweets
    """
    with open(file_path, 'r') as f:
        train_data = f.read().splitlines()
    return pd.DataFrame({ 'text' : train_data })


def read_test_data(file_path):
    """
    Reads test tweets.
    INPUT:
        file_path: path to file containing test set of tweets
    OUTPUT:
        data frame containing test set of tweets
    """
    with open(file_path, 'r') as f:
        test_data = f.read().splitlines()
    id_ = list(map(lambda x: x.split(',', 1)[0], test_data))
    text = list(map(lambda x: x.split(',', 1)[1], test_data))
    return pd.DataFrame({ 'id' : id_, 'text' : text })
    

if __name__ == "__main__":
    main()