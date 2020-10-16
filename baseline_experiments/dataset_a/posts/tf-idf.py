import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
from nltk.stem.porter import *
from filter_posts import prepro
from sentence_transformers import util


def tfidf_posts(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript, path_tweets, path_tweet_pics):
    fl, tw, connect = prepro(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics,
                             path_fl_descript, path_tweets, path_tweet_pics)

    # make results reproducible
    connect = connect.sort_values(by='twitterusername').reset_index(drop=True)

    # stem words
    ps = PorterStemmer()
    fl = [ps.stem(x) for x in fl]
    tw = [ps.stem(x) for x in tw]

    vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase=True, stop_words='english', max_df=0.7)
                                 #max_features=5000)
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
    print(f'tf-idf + cosine similarity - Dataset A')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':
    dataset = 'dataset_a'

    path_connect = f'../../../../data/{dataset}/connection.csv'
    path_fl_pics = f'../../../../data/{dataset}/flickr/flickr_pic_tags/'
    path_fl_descript = f'../../../../data/{dataset}/flickr/flickr_photo_descriptions/'
    path_tweets = f'../../../../data/{dataset}/twitter/tweets/'
    path_tweet_pics = f'../../../../data/{dataset}/twitter/tweet_pic_tags/'
    fl_keywords = True
    fl_descriptions = False
    tweet_keywords = False

    tfidf_posts(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript, path_tweets,
                path_tweet_pics)

    """
    with all data
    tf-idf + cosine similarity - Dataset A
    21 / 57
    0.3684210526315789
    
    without flickr keywords
    tf-idf + cosine similarity - Dataset A
    19 / 57
    0.3333333333333333
    
    without flickr description
    tf-idf + cosine similarity - Dataset A
    7 / 58
    0.1206896551724138
    
    without twitter keywords
    tf-idf + cosine similarity - Dataset A
    31 / 106
    0.29245283018867924
    
    without keywords
    tf-idf + cosine similarity - Dataset A
    31 / 108
    0.28703703703703703
    
    without fl_descriptions and without twitter keywords
    tf-idf + cosine similarity - Dataset A
    3 / 107
    0.028037383177570093
    
    """
