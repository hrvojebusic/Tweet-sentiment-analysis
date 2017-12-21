import numpy as np
import pandas as pd
import os
import csv
import re
import itertools
import time
import sys
from CNN import CNN
import tensorflow as tf
from hashtag_separator import get_word_vectors
from data_loader import load_glove_data


def trainCNN(tweets, train_tweets_ind, x_valid, y_valid, y_val, glove):    
    """
        Trains CNN
    INPUT:
        tweets: Dataframe with tweets 
        train_tweets_ind: shuffled indices of training dataset
        x_valid: tweets for validation
        y_valid: labels for tweets for validation
        y_val: matrix containing label for each tweet
        glove: glove dictionary
    OUTPUT: 
        path: path to the last saved checkpoint        
    """
    with tf.Graph().as_default():
        sess = tf.Session()
        
        with sess.as_default():
            cnn = CNN()
            global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.train.exponential_decay(5e-4, global_step, 500, 0.97, staircase=True, name='learning_rate')
            train_op = tf.train.AdamOptimizer(learning_rate, name='optimizer').minimize(cnn.loss, global_step=global_step, name='optim_operation')

            # Use timestamps for summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join("../data/models/", timestamp))
     
            # Loss, train and validation accuracy summaries for visualization
            loss_summary = tf.summary.scalar('loss', cnn.loss)
            acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)
            lambda_summary = tf.summary.scalar('learning_rate', learning_rate)
            
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, lambda_summary], name='training_summaries')
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            
            valid_summary_op = tf.summary.merge([loss_summary, acc_summary], name='validation_summaries')
            valid_summary_dir = os.path.join(out_dir, 'summaries', 'validation')
            valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

            # Checkpoint
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=50, name='saver')
            
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {cnn.x: x_batch, cnn.y: y_batch, cnn.dropout_prob: 0.5}
                _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                print('step %d,    loss %.3f,    accuracy %.2f' %(step,loss,100*accuracy))
                train_summary_writer.add_summary(summaries, step)
                
            def valid_step(x_batch, y_batch):
                feed_dict = {cnn.x: x_batch, cnn.y: y_batch, cnn.dropout_prob:1.0}
                step, summaries, loss, accuracy,  pred = sess.run([global_step, valid_summary_op, cnn.loss, cnn.accuracy, cnn.y_pred], feed_dict)
                print('step %d,    loss %.3f,    accuracy %.2f' %(step,loss,100*accuracy))
                valid_summary_writer.add_summary(summaries, step)
                
            for epoch in range(30):
                for batch_ind in batch_iter(train_tweets_ind, 1024):
                    minibatch_x = get_word_vectors(tweets.loc[batch_ind], glove)                    
                    minibatch_y = y_val[batch_ind, :]
                    
                    train_step(minibatch_x, minibatch_y)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % 20 == 0:
                        print("\nEvaluation:")
                        valid_step(x_valid, y_valid)
                    if current_step % 1000 == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                        
    return path


def batch_iter(train_tweets_ind, batch_size):
    """
	    Batch iterator
	INPUT:
		train_tweets_ind: indices for tweets to take in the batch
		batch_size: size of batch
	"""
    n_ind = len(train_tweets_ind)
    shuffled_indices = np.random.permutation(train_tweets_ind)
    for batch_num in range(int(np.ceil(n_ind/batch_size))):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, n_ind)
        if start_index != end_index:
            yield shuffled_indices[start_index:end_index]


def eval_from_checkpoint(test_tweets, path, glove):  
    """
        Evaluates predictions based on saved checkpoint
    INPUT:
        test_tweets: Dataframe with test tweets 
        path: location of checkpoint
        glove: glove dictionary
    OUTPUT: 
        test_predictions: predictions of test tweets     
    """   
    test_embeddings = get_word_vectors(test_tweets,  glove)
    
    graph = tf.Graph()
    with graph.as_default():
        checkpoint_file = tf.train.latest_checkpoint(path)
            
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            
            x = graph.get_operation_by_name("embedding").outputs[0]
            dropout_prob = graph.get_operation_by_name("dropout_prob").outputs[0]
            predictions = graph.get_operation_by_name("softmax/predicted_classes").outputs[0]
            
            test_predictions = sess.run(predictions, {x:test_embeddings,  dropout_prob:1.0})
            test_predictions[test_predictions==0] = -1
            
    return test_predictions

def create_submission(y_pred):
    """
        Creates submission file
    INPUT: 
            y_pred: list of predictions of test tweets 
    """
    with open("../output/submission.csv", 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        i = 1
        for y in y_pred:
            writer.writerow({'Id':int(i),'Prediction':y})
            i += 1

def main():
    if len(sys.argv) != 2 or (sys.argv[1] not in ['train', 'eval']):
        print("Invalid command. Expected 'train' or 'eval'.")
        return
    if sys.argv[1] == 'train':
        if not os.path.exists('../data/parsed/test_full.csv'):
            print('test_full.csv doesn\'t exist')
            return None
        tweets_test = pd.read_csv('../data/parsed/test_full.csv', names=['id', 'tweet'])
        tweets_test = tweets_test.drop(columns=['id'])


        if not os.path.exists('../data/parsed/train_pos_full.csv'):
            print('train_pos_full.csv doesn\'t exist')
            return None
        if not os.path.exists('../data/parsed/train_neg_full.csv'):
            print('train_neg_full.csv doesn\'t exist')
            return None
        pos = pd.read_csv('../data/parsed/train_pos_full.csv', names=['tweet'])
        pos['sentiment']=1
        neg = pd.read_csv('../data/parsed/train_neg_full.csv', names=['tweet'])
        neg['sentiment']=-1
        tweets_full = pd.concat([pos, neg], axis=0)
        tweets_full.reset_index(drop=True, inplace=True)

        
        print("Loading glove dictionary...")
        glove = load_glove_data()
        if glove == None:
            print("Failed loading glove dictionary!")
            return
        print("Dictionary loaded!")
        
        y_val = np.zeros((tweets_full.shape[0], 2))
        y_val[pos.shape[0]:, 0] = 1.0
        y_val[:pos.shape[0], 1] = 1.0
        shuffled = np.random.permutation(np.arange(tweets_full['tweet'].shape[0]))
        validation_tweets = shuffled[:1000]
        train_tweets_ind = shuffled[1000:]   
        x_valid = get_word_vectors(tweets_full.loc[validation_tweets], glove)
        y_valid = y_val[validation_tweets, :]
        path = trainCNN(tweets_full, train_tweets_ind, x_valid, y_valid, y_val, glove)

        pred = eval_from_checkpoint(tweets_test, path, glove)
        print('Creating submission file')
        create_submission(pred)
    if sys.argv[1] == 'eval':
        if not os.path.exists('../data/parsed/test_full.csv'):
            print('test_full.csv doesn\'t exist')
            return None
        tweets_test = pd.read_csv('../data/parsed/test_full.csv', names=['id', 'tweet'])
        tweets_test = tweets_test.drop(columns=['id'])
        
        print("Loading glove dictionary...")
        glove = load_glove_data()
        if glove == None:
            print("Failed loading glove dictionary!")
            return
        print("Dictionary loaded!")

        pred = eval_from_checkpoint(tweets_test, '../data/models/1513824111/checkpoints', glove)
        print('Creating submission file')
        create_submission(pred)
if __name__ == "__main__":
    main()