import numpy as np
import pandas as pd

from data_loader import load_glove_data
from os import path


def main():
    """
    Extracts all words from the GloVe word embeddings
    file and stores them separately for easier lookup.
    """
    print('Running GloVe word set extraction...')
    
    glove_dict = load_glove_data()
    glove_words = pd.DataFrame({ 'keys' : list(glove_dict.keys()) })
    
    keys_path = path.join('..', 'data', 'glove', 'glove_words.txt')
    glove_words.to_csv(keys_path, index=None, header=None)

    print('GloVe words saved to: {}'.format(keys_path))


if __name__ == '__main__':
    main()