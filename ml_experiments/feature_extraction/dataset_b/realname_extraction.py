import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import neighbors
from sklearn.svm import LinearSVC
from extract_funcs import get_ngrams
from extract_funcs import base_features
from extract_funcs import f_importances


def get_realnames(path_fl, path_tw, path_connect):
    """collects relevant realnames"""
    connect = pd.read_csv(path_connect, index_col=0)

    fl = pd.read_csv(path_fl, index_col=0)
    tw = pd.read_csv(path_tw, index_col=0, lineterminator='\n')

    fl = fl.dropna(subset=['realname', 'username'])
    tw = tw.dropna(subset=['name', 'username'])

    # only keep users where realname is available
    connect = connect[connect['flickrid'].isin(fl['nsid'])]
    connect = connect[connect['twitterusername'].isin(tw['username'])].reset_index(drop=True)

    # collect realnames in correct order
    fl_realnames = [fl.loc[fl['nsid'] == x, 'realname'].iloc[0] for x in connect['flickrid']]
    tw_realnames = [tw.loc[tw['username'] == x, 'name'].iloc[0] for x in connect['twitterusername']]

    return fl_realnames, tw_realnames, connect


def get_base_features(train_realnames, test_realnames):
    """calculates base features"""

    # feature names begin with 'real_' to distinguish from other features
    base_feature_names = ['real_numbers', 'real_letters', 'real_spaces', 'real_other_chars', 'real_length',
                          'real_number_perc', 'real_letter_perc']

    # calculate features for training data
    train_features = [base_features(x) for x in train_realnames]
    train_features = pd.DataFrame(train_features, columns=base_feature_names)

    # calculate features for test data
    test_features = [base_features(x) for x in test_realnames]
    test_features = pd.DataFrame(test_features, columns=base_feature_names)
    return train_features, test_features


def execute(path_fl, path_tw, path_connect, min_gram, max_gram, min_df, max_features, analyzer, path_to_train, path_to_test, flickr,
            csv):
    """extract features and write to csv"""
    fl_realnames, tw_realnames, connect = get_realnames(path_fl, path_tw, path_connect)

    if flickr:
        X_train_grams, X_test_grams = get_ngrams(fl_realnames, tw_realnames, min_gram, max_gram, min_df, max_features, analyzer, 'real_')
        X_train_base, X_test_base = get_base_features(fl_realnames, tw_realnames)
    else:
        X_train_grams, X_test_grams = get_ngrams(tw_realnames, fl_realnames, min_gram, max_gram, min_df, max_features, analyzer, 'real_')
        X_train_base, X_test_base = get_base_features(tw_realnames, fl_realnames)

    X_train = X_train_grams
    X_test = X_test_grams

    X_train = pd.concat([X_train_grams, X_train_base], axis=1)
    X_test = pd.concat([X_test_grams, X_test_base], axis=1)

    fl_username = connect['flickrusername']
    tw_username = connect['twitterusername']

    res = []
    for i in range(len(fl_username)):
        res.append(f'{fl_username[i]} + {tw_username[i]}')

    X_train['label'] = pd.Series(res)
    X_test['label'] = pd.Series(res)

    if csv:
        X_train.to_csv(path_to_train)
        X_test.to_csv(path_to_test)

    # -------------------------------------

    """

    X_train = X_train.sample(frac=1)
    X_test = X_test.sample(frac=1)

    y_train = X_train.pop('label')
    y_test = X_test.pop('label')

    print('realname features')
    print(X_train.shape)

    feature_names = X_train.columns

    clf = LinearSVC(max_iter=5000)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    f_importances(abs(clf.coef_[0]), feature_names, top=30)

    # clf = neighbors.KNeighborsClassifier(1, weights='uniform')
    # clf.fit(X_fl, y_fl)
    # print(clf.score(X_tw, y_tw))
    """


if __name__ == '__main__':
    flickr = False  # TRUE -> ngrams learned from flickr, FALSE -> ngrams learned from twitter
    csv = False
    min_gram = 1
    max_gram = 3
    min_df = 10  # 10
    max_features = None
    analyzer = 'char'
    init_voc = 'flickr' if flickr else 'twitter'
    dataset = 'dataset_b'

    path_fl = f'../../../../data/{dataset}/flickr/flickr_info.csv'
    path_tw = f'../../../../data/{dataset}/twitter/twitter_info.csv'
    path_connect = f'../../../../data/{dataset}/connection.csv'
    path_to_train = f'../../features/dataset_b/voc_learned_{init_voc}/realname_train.csv'
    path_to_test = f'../../features/dataset_b/voc_learned_{init_voc}/realname_test.csv'

    execute(path_fl, path_tw, path_connect, min_gram, max_gram, min_df, max_features, analyzer, path_to_train, path_to_test,
            flickr, csv)



