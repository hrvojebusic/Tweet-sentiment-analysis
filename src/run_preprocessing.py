import sys
import pandas as pd

from os import path
from tweet_preprocessing import preprocess_tweets


def main():
    if len(sys.argv) != 2 or (sys.argv[1] not in ['train', 'test']):
        print("Invalid command. Expected 'train' or 'test'.")
        return

    if sys.argv[1] == 'train':
        process_train_data()
    if sys.argv[1] == 'test':
        process_test_data()
    

def process_train_data():
    path_read_pos_full = path.join('..', 'data', 'dataset', 'train_pos_full.txt')
    print('Reading positive tweets from: {}'.format(path_read_pos_full))
    with open(path_read_pos_full, 'r') as f:
        pos_full = f.read().splitlines()
    print('Read positive tweets: {}'.format(len(pos_full)))
    pos_df = pd.DataFrame({ 'text' : pos_full })

    path_read_neg_full = path.join('..', 'data', 'dataset', 'train_neg_full.txt')
    print('Reading negative tweets from: {}'.format(path_read_neg_full))
    with open(path_read_neg_full, 'r') as f:
        neg_full = f.read().splitlines()
    print('Read negative tweets: {}'.format(len(neg_full)))
    neg_df = pd.DataFrame({ 'text' : neg_full })

    print('Processing positive tweets...')
    pos_df = preprocess_tweets(pos_df, 'text')

    print('Processing negative tweets...')
    neg_df = preprocess_tweets(neg_df, 'text')

    path_save_pos_full = path.join('..', 'data', 'parsed', 'train_pos_full.csv')
    print('Saving processed positive tweets to: {}'.format(path_save_pos_full))
    pos_df.to_csv(path_save_pos_full, header=False, index=False)

    path_save_neg_full = path.join('..', 'data', 'parsed', 'train_neg_full.csv')
    print('Saving processed negative tweets to: {}'.format(path_save_neg_full))
    neg_df.to_csv(path_save_neg_full, header=False, index=False)


def process_test_data():
    path_read_test = path.join('..', 'data', 'dataset', 'test_data.txt')
    print('Reading test tweets from: {}'.format(path_read_test))
    with open(path_read_test, 'r') as f:
        test_full = f.read().splitlines()
    print('Read tweets: {}'.format(len(test_full)))

    print('Processing test tweets...')
    id_ = list(map(lambda x: x.split(',', 1)[0], test_full))
    text = list(map(lambda x: x.split(',', 1)[1], test_full))
    test_df = pd.DataFrame({ 'id' : id_, 'text' : text })

    parameters = {
            'filter_duplicates' : False,
            'remove_user_tags' : True,
            'remove_url_tags' : True,
            'replace_emoticons' : True,
            'split_hashtags' : True,
            'emphasize_punctuation': True,
            'remove_small_words': True,
            'remove_non_chars': True,
            'check_tweet_emphasis' : True,
            'remove_numbers' : True,
            'remove_stopwords' : True,
            'infer_sentiment' : True,
            'stem' : False,
            'lemmatize' : False
        }

    test_df.text = preprocess_tweets(test_df, 'text', parameters)

    path_save_train = path.join('..', 'data', 'parsed', 'test_full.csv')
    print('Saving processed test tweets to: {}'.format(path_save_train))
    test_df.to_csv(path_save_train, header=False, index=False)


if __name__ == "__main__":
    main()