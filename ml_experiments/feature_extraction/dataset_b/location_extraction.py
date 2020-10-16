import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import neighbors
from sklearn.svm import LinearSVC
from extract_funcs import get_ngrams
from extract_funcs import base_features
from extract_funcs import f_importances


def get_locations(path_fl, path_tw, path_connect):
    """collects relevant locations"""
    connect = pd.read_csv(path_connect, index_col=0)

    fl = pd.read_csv(path_fl, index_col=0)
    tw = pd.read_csv(path_tw, index_col=0, lineterminator='\n')

    fl = fl.dropna(subset=['location'])
    tw = tw.dropna(subset=['location'])

    # only keep users where location is available
    connect = connect[connect['flickrid'].isin(fl['nsid'])]
    connect = connect[connect['twitterusername'].isin(tw['username'])].reset_index(drop=True)

    # collect locations in correct order
    fl_locations = [fl.loc[fl['nsid'] == x, 'location'].iloc[0] for x in connect['flickrid']]
    tw_locations = [tw.loc[tw['username'] == x, 'location'].iloc[0] for x in connect['twitterusername']]

    return fl_locations, tw_locations, connect


def get_base_features(train_realnames, test_realnames):
    """calculates base features"""

    # feature names begin with 'loc_' to distinguish from other features
    base_feature_names = ['loc_numbers', 'loc_letters', 'loc_spaces', 'loc_other_chars', 'loc_length',
                          'loc_number_perc', 'loc_letter_perc']

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
    fl_locations, tw_locations, connect = get_locations(path_fl, path_tw, path_connect)

    if flickr:
        X_train_grams, X_test_grams = get_ngrams(fl_locations, tw_locations, min_gram, max_gram, min_df, max_features, analyzer, 'loc_')
        X_train_base, X_test_base = get_base_features(fl_locations, tw_locations)
    else:
        X_train_grams, X_test_grams = get_ngrams(tw_locations, fl_locations, min_gram, max_gram, min_df, max_features, analyzer, 'loc_')
        X_train_base, X_test_base = get_base_features(tw_locations, fl_locations)

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
    path_to_train = f'../../features/dataset_b/voc_learned_{init_voc}/location_train.csv'
    path_to_test = f'../../features/dataset_b/voc_learned_{init_voc}/location_test.csv'

    execute(path_fl, path_tw, path_connect, min_gram, max_gram, min_df, max_features, analyzer, path_to_train,
            path_to_test, flickr, csv)



