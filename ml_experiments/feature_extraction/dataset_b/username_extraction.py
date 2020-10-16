import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import neighbors
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from extract_funcs import get_ngrams
from extract_funcs import base_features
from extract_funcs import f_importances


def get_names(path_connect):
    """collects relevant usernames"""
    connect = pd.read_csv(path_connect, index_col=0)

    # drop missing values
    connect = connect.dropna(subset=['flickrusername', 'twitterusername']).reset_index(drop=True)

    return connect['flickrusername'], connect['twitterusername']


def get_base_features(train_usernames, test_usernames):
    """calculates base features"""

    # feature names begin with 'user_' to distinguish from other features
    base_feature_names = ['user_numbers', 'user_letters', 'user_spaces', 'user_other_chars', 'user_length',
                          'user_number_perc', 'user_letter_perc']

    # calculate features for training data
    train_features = [base_features(x) for x in train_usernames]
    train_features = pd.DataFrame(train_features, columns=base_feature_names)

    # calculate features for test data
    test_features = [base_features(x) for x in test_usernames]
    test_features = pd.DataFrame(test_features, columns=base_feature_names)

    return train_features, test_features


def execute(path_connect, min_gram, max_gram, min_df, max_features, analyzer, path_to_train, path_to_test, flickr, csv):
    """extract features and write to csv"""

    fl_username, tw_username = get_names(path_connect)

    if flickr:
        X_train_grams, X_test_grams = get_ngrams(fl_username, tw_username, min_gram, max_gram, min_df, max_features, analyzer, 'user_')
        X_train_base, X_test_base = get_base_features(fl_username, tw_username)
    else:
        X_train_grams, X_test_grams = get_ngrams(tw_username, fl_username, min_gram, max_gram, min_df, max_features, analyzer, 'user_')
        X_train_base, X_test_base = get_base_features(tw_username, fl_username)

    X_train = X_train_grams
    X_test = X_test_grams

    X_train = pd.concat([X_train_grams, X_train_base], axis=1)
    X_test = pd.concat([X_test_grams, X_test_base], axis=1)

    res = []
    for i in range(len(fl_username)):
        res.append(f'{fl_username[i]} + {tw_username[i]}')

    X_train['label'] = pd.Series(res)
    X_test['label'] = pd.Series(res)

    if csv:
        X_train.to_csv(path_to_train)
        X_test.to_csv(path_to_test)

    # -------------------------------------

    X_train = X_train.sample(frac=1)
    X_test = X_test.sample(frac=1)

    print(X_train.shape)

    y_train = X_train.pop('label')
    y_test = X_test.pop('label')

    feature_names = X_train.columns

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #clf = neighbors.KNeighborsClassifier(1, weights='uniform')
    #clf.fit(X_fl, y_fl)
    #print(clf.score(X_tw, y_tw))

    clf = LinearSVC(random_state=0, max_iter=5000)
    clf.fit(X_train, y_train)
    print('-------')
    print(clf.score(X_test, y_test))
    print('-------')


    f_importances(abs(clf.coef_[0]), feature_names, top=30)


if __name__ == '__main__':

    csv = False
    flickr = False  # TRUE -> ngrams learned from flickr, FALSE -> ngrams learned from twitter

    min_gram = 1
    max_gram = 3
    min_df = 10  #10
    max_features = None
    analyzer = 'char_wb'
    init_voc = 'flickr' if flickr else 'twitter'
    dataset = 'dataset_b'
    path_connect = f'../../../../data/{dataset}/connection.csv'
    path_to_train = f'../../features/dataset_b/voc_learned_{init_voc}/username_train.csv'
    path_to_test = f'../../features/dataset_b/voc_learned_{init_voc}/username_test.csv'

    execute(path_connect, min_gram, max_gram, min_df, max_features, analyzer, path_to_train, path_to_test, flickr, csv)


    """ new
    from flickr
    (5924, 1000 + base)
    0.675388251181634
    
    from twitter
    (5924, 1000 + base)
    0.6900742741390952
    """




    """ old
    (5924, 2080)
    0.6978392977717758
    """







