from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import util
import pandas as pd
import torch
import numpy as np
from nltk.stem.porter import *
from sklearn.svm import LinearSVC
from extract_funcs import get_tfidf



def tfidf_bio(path_connection, max_df, max_features, min_df, flickr, csv):
    """extract bio features using tf-idf vectors"""

    # read data
    df = pd.read_csv(path_connection, index_col=0, lineterminator='\n')

    # make results reproducible
    df = df.sort_values(by='twitterusername').reset_index(drop=True)

    # extract usernames
    fl_names = df['flickrusername'].tolist()
    tw_names = df['twitterusername'].tolist()

    # extract bios
    fl_bio = df['flickr_bio'].tolist()
    tw_bio = df['twitter_bio'].tolist()

    # stem words
    ps = PorterStemmer()
    fl_bio = [ps.stem(x) for x in fl_bio]
    tw_bio = [ps.stem(x) for x in tw_bio]

    if flickr:
        X_train, X_test = get_tfidf(fl_bio, tw_bio, max_df, max_features, min_df, 'bio_')
    else:
        X_train, X_test = get_tfidf(tw_bio, fl_bio, max_df, max_features, min_df, 'bio_')

    fl_username = df['flickrusername']
    tw_username = df['twitterusername']

    res = []
    for i in range(len(fl_username)):
        res.append(f'{fl_username[i]} + {tw_username[i]}')

    X_train['label'] = pd.Series(res)
    X_test['label'] = pd.Series(res)

    if csv:
        X_train.to_csv(path_to_train)
        X_test.to_csv(path_to_test)

    X_train = X_train.sample(frac=1)
    X_test = X_test.sample(frac=1)

    y_train = X_train.pop('label')
    y_test = X_test.pop('label')

    clf = LinearSVC(max_iter=10000)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


if __name__ == '__main__':

    flickr = False
    csv = True
    dataset = 'dataset_b'

    max_df = 0.7
    max_features = 20000
    min_df = 1

    init_voc = 'flickr' if flickr else 'twitter'
    path_to_train = f'../../features/dataset_b/voc_learned_{init_voc}/bio_train.csv'
    path_to_test = f'../../features/dataset_b/voc_learned_{init_voc}/bio_test.csv'
    path_connection = f'../../../baseline_experiments/{dataset}/biography/bio_connection_{dataset}.csv'

    tfidf_bio(path_connection, max_df, max_features, min_df, flickr, csv)

    """ Flickr = False
    (2880, 10459)
    0.14131944444444444
    """

