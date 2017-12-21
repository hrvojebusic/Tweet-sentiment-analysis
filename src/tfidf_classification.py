import data_loader
import numpy as np
import pandas as pd
import sys
import time

from os import path
from pickle import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['svm', 'nn']:
        print("Invalid command. Expected 'svm' or 'nn'.")
        return

    c_name = sys.argv[1]
    print('Running job: TF-IDF vectorization and ' + c_name.upper() + ' classifier.')

    train_data = data_loader.load_train_data().sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = data_loader.load_test_data()

    if c_name == 'svm':
        classifier = LinearSVC(random_state=42)
        param_grid = {
            'vectorizer__ngram_range' : [(1,1), (1,2), (1,3), (1,4)],
            'classifier__C' : [0.1, 1]
        }
    else:
        classifier = MLPClassifier((50,), solver='lbfgs', learning_rate_init=1e-4, tol=1e-6, max_iter=200, random_state=42)
        param_grid = {
            'vectorizer__ngram_range' : [(1,1)]
        }

    pipe = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', classifier)
    ])

    cv_grid = GridSearchCV(pipe, n_jobs=2, cv=5, verbose=3, param_grid=param_grid)
    
    start_time = time.time()
    cv_grid.fit(train_data.text, train_data.sentiment)
    end_time = time.time()
    print('Total fit time: {}'.format(end_time - start_time))

    # Classification report
    pred = cv_grid.predict(train_data.text)
    cr = classification_report(train_data.sentiment, pred)
    print(cr)

    # Test predictions
    pred = cv_grid.predict(test_data.text)
    print('Predictions finished.')

    # Save predictions
    results = pd.DataFrame({ 'Id': test_data.index, 'Prediction': pred })
    results = results.set_index('Id')
    data_loader.save_submission(results, 'tfidf_' + c_name.upper() + '_submission.csv')
    print('Predictions saved.')

    # Save classification results
    cvr_path = path.join('pickles', 'tfidf_' + c_name.upper() + '_cross_validation_results') # Cross validation results
    be_path = path.join('pickles', 'tfidf_' + c_name.upper() + '_best_estimator') # Best estimator

    dump(cv_grid.cv_results_, open(cvr_path, 'wb'))
    dump(cv_grid.best_estimator_, open(be_path, 'wb'))
    print('Classification results saved.')


if __name__ == '__main__':
    main()