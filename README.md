# Preprocessed tweet files
Go to our github project [page](https://github.com/dinomujki/project2) to download necessary files!!!

# Tweet Sentiment Analysis

The competition task was to predict if a tweet message used to contain a positive :) or a negative :( smiley, by considering only the remaining text. Our team conducted comprehensive research on the proposed solutions in the relevant literature, as well as past projects and articles which tackled similar issues regarding text sentiment analysis. Full specification of our experiments, as well as results and conclusions drawn can be found in our report.

Complete project specification is available on the course's [GitHub page](https://github.com/epfml/ML_course/tree/master/projects/project2/project_text_classification).

## Dependencies

Following dependencies are required in order to run the project:

### Libraries

* [Anaconda3](https://www.anaconda.com/download/) - Download and install Anaconda with Python3

* [Scikit-Learn](http://scikit-learn.org/stable/install.html) - Download scikit-learn library with conda
    ```sh
    conda install scikit-learn
    ```

* [Gensim](https://radimrehurek.com/gensim/) - Install Gensim library
    ```sh
    conda install gensim
    ```

* [NLTK](http://www.nltk.org/data.html) - Download all the packages of NLTK
    ```sh
    python
    >>> import nltk
    >>> nltk.download()
    ```
  
* [Tensorflow](https://www.tensorflow.org/get_started/os_setup) - Install tensorflow library (version used **1.4.1**)

    ```sh
    $ pip install tensorflow
    ```

### Files

* Train tweets

    [Download](https://www.kaggle.com/c/epfml17-text/data) `twitter-datasets.zip` containing positive and negative tweet files which are required during the model training phase. After unzipping, place the files obtained in the `./data/datasets` directory.

* Test tweets

    [Download](https://www.kaggle.com/c/epfml17-text/data) `test_data.txt` containing tweets which are required for the testing of the trained model and obtaining score for submission to Kaggle. This file needs to be placed in the `./data/datasets` directory.

* Stanford Pretrained Glove Word Embeddings

    [Download](http://nlp.stanford.edu/data/glove.twitter.27B.zip) *Glove Pretrained Word Embeddings* which are used for training advanced sentiment analysis models. After unzipping, place the file `glove.twitter.27B.200d.txt` in the `./data/glove` directory.

## Hardware requirements

* at least **16 GB** of RAM
* a **graphics card** (optional for faster training involving CNNs)

Tested on Ubuntu 16.04 with Nvidia Tesla K80 GPU with 12 GB GDDR5

## Kaggle competition

[Public Leaderboard](https://www.kaggle.com/c/epfml17-text/leaderboard) connected to this competition.

Our team's name is **Bill Trader**.

Team members:

* Dino Mujkić ([dinomujki](https://github.com/dinomujki))
* Hrvoje Bušić ([hrvojebusic](https://github.com/hrvojebusic))
* Sebastijan Stevanović ([sebastijan94](https://github.com/sebastijan94))

## Reproducing our best result
You can find already preprocessed tweet files `test_full.csv`, `train_neg_full.csv.zip` and `train_pos_full.csv.zip` in the `./data/parsed` directory.

To run preprocessing again you must have Train tweets and Test tweets files in the `./data/dataset` directory. Then go to folder `/src` and run `run_preprocessing.py` with argument `train` or `test` to generate requried files for running the CNN.

```sh
$ python run_preprocessing.py train or test
```

To reproduce our best score from Kaggle go to folder `/src` and run `run_cnn.py` with argument `eval`

```sh
$ python run_cnn.py eval
```

In `data/models/1513824111` directory is stored a checkpoint for reproducing our best score so the training part will be skipped. If you want to run the training process from scratch, just pass the argument `train` when runnig `run_cnn.py`.

**To run the evaluation you must have the necessary files. File `glove.twitter.27B.200d.txt` in the `./data/glove` directory and preprocessed tweet files `test_full.csv`, `train_neg_full.csv` and `train_pos_full.csv` in the `./data/parsed` directory.**
___

This project is available under [MIT](https://opensource.org/licenses/MIT) license.
