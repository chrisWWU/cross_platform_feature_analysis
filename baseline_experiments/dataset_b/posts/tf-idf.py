import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
from nltk.stem.porter import *
from filter_posts import prepro
from sentence_transformers import util


def tfidf_posts(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript, path_tweets,
                path_tweet_pics):
    fl, tw, connect = prepro(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript,
                             path_tweets, path_tweet_pics)

    # make results reproducible
    connect = connect.sort_values(by='twitterusername').reset_index(drop=True)

    # stem words
    ps = PorterStemmer()
    fl = [ps.stem(x) for x in fl]
    tw = [ps.stem(x) for x in tw]

    vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase=True, stop_words='english', max_df=0.7)
                                 #max_features=10000)
    vector_fl = vectorizer.fit_transform(fl)
    vector_tw = vectorizer.transform(tw)

    # convert to tensor for faster cosine calculation
    fl_emb = torch.tensor(vector_fl.toarray())
    tw_emb = torch.tensor(vector_tw.toarray())

    # Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(fl_emb, tw_emb)
    cosine_scores[cosine_scores != cosine_scores] = 0

    # get indices of max similarity values
    ind = torch.argmax(cosine_scores, dim=1)

    # extract usernames
    fl_names = connect['flickrusername'].tolist()
    tw_names = connect['twitterusername'].tolist()

    # initialize result
    result = pd.DataFrame(columns=['flickrusername', 'twitterusername'])

    for i in range(len(fl)):
        result.loc[i] = [fl_names[i], tw_names[ind[i]]]

    # compare actual matching with tfidf matching
    df1 = result.merge(connect, on=['flickrusername', 'twitterusername'], how='left', indicator='Exist')
    df1['Exist'] = np.where(df1.Exist == 'both', True, False)

    # count correct matches
    cor = df1['Exist'].sum()

    # print performance measures
    print(f'tf-idf + cosine similarity - Dataset B')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':
    dataset = 'dataset_b'

    path_connect = f'../../../../data/{dataset}/connection.csv'
    path_fl_pics = f'../../../../data/{dataset}/flickr/flickr_pic_tags/'
    path_fl_descript = f'../../../../data/{dataset}/flickr/flickr_photo_descriptions/'
    path_tweets = f'../../../../data/{dataset}/twitter/tweets/'
    path_tweet_pics = f'../../../../data/{dataset}/twitter/tweet_pic_tags/'
    # decide which data to consider (if both fl options are false fl_descriptions are utilized)
    fl_keywords = True
    fl_descriptions = False
    # tweet text is always used, additionally tweet keywords can be considered
    tweet_keywords = False

    tfidf_posts(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript, path_tweets,
                path_tweet_pics)


    """ unlimited features:
    with all data
    tf-idf + cosine similarity - Dataset B
    254 / 2301
    0.11038678835289005
    
    without flickr keywords
    tf-idf + cosine similarity - Dataset B
    259 / 2445
    0.10593047034764826
    
    without flickr description
    tf-idf + cosine similarity - Dataset B
    22 / 2325
    0.00946236559139785
    
    without twitter keywords
    tf-idf + cosine similarity - Dataset B
    413 / 4931
    0.08375583046035287
    
    without keywords
    tf-idf + cosine similarity - Dataset B
    417 / 5233
    0.07968660424230843
    
    without fl_descriptions and without twitter keywords
    tf-idf + cosine similarity - Dataset B
    15 / 4983
    0.0030102347983142685
    
    """