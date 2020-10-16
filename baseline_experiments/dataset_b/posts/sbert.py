import pandas as pd
import numpy as np
import torch
from filter_posts import prepro
from sentence_transformers import SentenceTransformer, util


def sbert_posts(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript, path_tweets,
                path_tweet_pics, model):
    fl, tw, connect = prepro(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics,
                             path_fl_descript, path_tweets, path_tweet_pics)

    # make results reproducible
    connect = connect.sort_values(by='twitterusername').reset_index(drop=True)

    # Compute embedding for both lists
    fl_sentence_embeddings = model.encode(fl, convert_to_tensor=True)
    tw_sentence_embeddings = model.encode(tw, convert_to_tensor=True)

    # Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(fl_sentence_embeddings, tw_sentence_embeddings)
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
    print(f'sbert + cosine similarity - Dataset B')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':
    dataset = 'dataset_b'
    path_connect = f'../../../../data/{dataset}/connection.csv'
    path_fl_pics = f'../../../../data/{dataset}/flickr/flickr_pic_tags/'
    path_fl_descript = f'../../../../data/{dataset}/flickr/flickr_photo_descriptions/'
    path_tweets = f'../../../../data/{dataset}/twitter/tweets/'
    path_tweet_pics = f'../../../../data/{dataset}/twitter/tweet_pic_tags/'
    model_names = ['bert-base-nli-mean-tokens', 'distilbert-base-nli-stsb-mean-tokens']
    model = SentenceTransformer(model_names[0])  # 0 or 1
    fl_keywords = True  # decide which data to consider (if both fl options are false fl_descriptions are utilized)
    fl_descriptions = False  # tweet text is always used, additionally tweet keywords can be considered
    tweet_keywords = False

    sbert_posts(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript, path_tweets,
                path_tweet_pics, model)

    """ bert-base-nli-mean-tokens
    with all data
    sbert + cosine similarity - Dataset B
    26 / 2301
    0.011299435028248588

    without flickr keywords
    sbert + cosine similarity - Dataset B
    27 / 2445
    0.011042944785276074

    without flickr description
    sbert + cosine similarity - Dataset B
    7 / 2325
    0.003010752688172043

    without twitter keywords
    sbert + cosine similarity - Dataset B
    35 / 4931
    0.007097951733928209

    without keywords
    sbert + cosine similarity - Dataset B
    36 / 5233
    0.006879419071278425

    without fl_descriptions and without twitter keywords
    sbert + cosine similarity - Dataset B
    9 / 4983
    0.001806140878988561
    """

    """ distilbert-base-nli-stsb-mean-tokens
        with all data
    sbert + cosine similarity - Dataset B
    28 / 2301
    0.012168622338113864

    without flickr keywords
    sbert + cosine similarity - Dataset B
    30 / 2445
    0.012269938650306749

    without flickr description
    sbert + cosine similarity - Dataset B
    10 / 2325
    0.004301075268817204

    without twitter keywords
    sbert + cosine similarity - Dataset B
    34 / 4931
    0.006895153112958832

    without keywords
    sbert + cosine similarity - Dataset B
    37 / 5233
    0.007070514045480604

    without fl_descriptions and without twitter keywords
    sbert + cosine similarity - Dataset B
    7 / 4983
    0.0014047762392133253
    """