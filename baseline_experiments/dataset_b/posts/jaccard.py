import pandas as pd
import numpy as np
from nltk.stem.porter import *
from filter_posts import prepro

def simJaccard(str1, str2):
    """calculate jaccard similarity"""
    str1 = set(str1.split())
    str2 = set(str2.split())
    if len(str1 | str2) == 0:
        return 0
    else:
        return float(len(str1 & str2)) / len(str1 | str2)


def jaccard_posts(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript,
                  path_tweets, path_tweet_pics):
    fl, tw, connect = prepro(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics,
                             path_fl_descript, path_tweets, path_tweet_pics)

    # stem words
    ps = PorterStemmer()
    fl = [ps.stem(x) for x in fl]
    tw = [ps.stem(x) for x in tw]

    connect = pd.read_csv(path_connect, index_col=0)
    fl_names = connect['flickrusername']
    tw_names = connect['twitterusername']

    # initialize result
    result = pd.DataFrame(columns=['flickrusername', 'twitterusername'])
    posts = pd.DataFrame(columns=['flickrposts', 'twitterposts'])
    c = 0
    for i in fl:
        print(f'{c+1} / {len(fl)}')
        # get similarity for each combination
        res = [simJaccard(i, j) for j in tw]

        # select flickr user from current iteration
        fl_match = fl_names[c]

        # select corresponding twitter user with max similarity
        index_max = np.argmax(res)
        match = tw_names[index_max]

        # append pair to result
        result.loc[c] = [fl_match, match]

        # just to show bio matches:
        posts.loc[c] = [i, tw[index_max]]

        c += 1

    #print(posts)

    # compare actual matching with jaccard matching
    df1 = result.merge(connect, on=['flickrusername', 'twitterusername'], how='left', indicator='Exist')
    df1['Exist'] = np.where(df1.Exist == 'both', True, False)

    # count correct matches
    cor = df1['Exist'].sum()

    # print performance measures
    print('Jaccard - Dataset B')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':
    path_connect = '/Users/kiki/Desktop/casia_cross_osn_local_data_IMPORTANT/7_combined_connection.csv'
    path_fl_pics = '/Users/kiki/Desktop/casia_cross_osn_local_data_IMPORTANT/flickr/flickr_pic_tags_cross_osn/'
    path_fl_descript = '/Users/kiki/Desktop/casia_cross_osn_local_data_IMPORTANT/flickr/photo_descriptions/'
    path_tweets = '/Users/kiki/Desktop/casia_cross_osn_local_data_IMPORTANT/twitter/tweets/'
    path_tweet_pics = '/Users/kiki/Desktop/casia_cross_osn_local_data_IMPORTANT/twitter/pred_tweet_pics_cross_osn/'
    fl_keywords = False  # decide which data to consider (if both fl options are false fl_descriptions are utilized)
    fl_descriptions = True  # tweet text is always used, additionally tweet keywords can be considered
    tweet_keywords = True

    jaccard_posts(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript,
                  path_tweets,path_tweet_pics)

    """
    with all data with stemming
    Jaccard - Dataset B
    185 / 2301
    0.08039982616253803

    without flickr keywords with stemming
    Jaccard - Dataset B
    186 / 2445
    0.07607361963190185

    without flickr description with stemming
    tf-idf + cosine similarity - Dataset B
    22 / 2325
    0.00946236559139785

    without twitter keywords with stemming
    tf-idf + cosine similarity - Dataset B
    216 / 4931
    0.04380450212938552

    without keywords with stemming
    tf-idf + cosine similarity - Dataset B
    215 / 5233
    0.04108541945346837

    without fl_descriptions and without twitter keywords with stemming
    tf-idf + cosine similarity - Dataset B
    15 / 4983
    0.0030102347983142685

    """