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

    # make results reproducible
    connect = connect.sort_values(by='twitterusername').reset_index(drop=True)

    # stem words
    ps = PorterStemmer()
    fl = [ps.stem(x) for x in fl]
    tw = [ps.stem(x) for x in tw]

    # make results reproducible
    connect = connect.sort_values(by='twitterusername').reset_index(drop=True)

    fl_names = connect['flickrusername']
    tw_names = connect['twitterusername']

    # initialize result
    result = pd.DataFrame(columns=['flickrusername', 'twitterusername'])
    posts = pd.DataFrame(columns=['flickrposts', 'twitterposts'])
    c = 0
    for i in fl:
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
    print('Jaccard - Dataset A')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':

    dataset = 'dataset_a'

    path_connect = f'../../../../data/{dataset}/connection.csv'
    path_fl_pics = f'../../../../data/{dataset}/flickr/flickr_pic_tags/'
    path_fl_descript = f'../../../../data/{dataset}/flickr/flickr_photo_descriptions/'
    path_tweets = f'../../../../data/{dataset}/twitter/tweets/'
    path_tweet_pics = f'../../../../data/{dataset}/twitter/tweet_pic_tags/'
    fl_keywords = False
    fl_descriptions = True
    tweet_keywords = False

    jaccard_posts(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript,
                  path_tweets, path_tweet_pics)

    """
    with all data
    Jaccard - Dataset A
    20 / 57
    0.3508771929824561

    without flickr keywords
    Jaccard - Dataset A
    17 / 57
    0.2982456140350877

    without flickr description
    Jaccard - Dataset A
    5 / 58
    0.08620689655172414

    without twitter keywords
    Jaccard - Dataset A
    29 / 106
    0.27358490566037735

    without keywords
    Jaccard - Dataset A
    28 / 108
    0.25925925925925924

    without fl_descriptions and without twitter keywords
    Jaccard - Dataset A
    6 / 107
    0.056074766355140186

    """
