from baseline_experiments.dataset_b.posts.filter_posts import prepro
from extract_funcs import get_tfidf
from nltk.stem.porter import *
from sklearn.svm import LinearSVC
import pandas as pd


def tfidf_posts(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript,
                path_tweets, path_tweet_pics, max_df, max_features, min_df, flickr, csv):
    """creates post features using tf-idf vectors"""

    # prepro function bundles the post data (include keywords or not)
    fl, tw, connect = prepro(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics,
                             path_fl_descript, path_tweets, path_tweet_pics)

    # stem words
    ps = PorterStemmer()
    fl = [ps.stem(x) for x in fl]
    tw = [ps.stem(x) for x in tw]

    if flickr:
        X_train, X_test = get_tfidf(fl, tw, max_df, max_features, min_df, 'post_')
    else:
        X_train, X_test = get_tfidf(tw, fl, max_df, max_features, min_df, 'post_')

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

    # ------------------------- delete below

    X_train = X_train.sample(frac=1, random_state=0)
    X_test = X_test.sample(frac=1, random_state=0)

    y_train = X_train.pop('label')
    y_test = X_test.pop('label')

    clf = LinearSVC(max_iter=5000, dual=True)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


if __name__ == '__main__':
    flickr = False
    csv = False
    max_df = 0.7
    max_features = 20000
    min_df = 1

    init_voc = 'flickr' if flickr else 'twitter'
    path_to_train = f'../../features/combined_datasets/voc_learned_{init_voc}/post_train.csv'
    path_to_test = f'../../features/combined_datasets/voc_learned_{init_voc}/post_test.csv'

    dataset = 'datasets_combined'

    path_connect = f'../../../../data/{dataset}/connection.csv'
    path_fl_pics = f'../../../../data/{dataset}/flickr/flickr_pic_tags/'
    path_fl_descript = f'../../../../data/{dataset}/flickr/flickr_photo_descriptions/'
    path_tweets = f'../../../../data/{dataset}/twitter/tweets/'
    path_tweet_pics = f'../../../../data/{dataset}/twitter/tweet_pic_tags/'

    fl_keywords = False  # decide which data to consider (if both fl options are false fl_descriptions are utilized)
    fl_descriptions = True  # tweet text is always used, additionally tweet keywords can be considered
    tweet_keywords = False

    tfidf_posts(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript,
                path_tweets, path_tweet_pics, max_df, max_features, min_df, flickr, csv)

