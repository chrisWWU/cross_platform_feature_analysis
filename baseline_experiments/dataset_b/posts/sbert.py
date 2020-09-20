import pandas as pd
import numpy as np
import torch
from filter_posts import prepro
from sentence_transformers import SentenceTransformer, util


def sbert_posts(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript, path_tweets,
                path_tweet_pics, model):
    fl, tw, connect = prepro(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics,
                             path_fl_descript, path_tweets, path_tweet_pics)

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
    path_connect = '/Users/kiki/Desktop/casia_cross_osn_local_data_IMPORTANT/7_combined_connection.csv'
    path_fl_pics = '/Users/kiki/Desktop/casia_cross_osn_local_data_IMPORTANT/flickr/flickr_pic_tags_cross_osn/'
    path_fl_descript = '/Users/kiki/Desktop/casia_cross_osn_local_data_IMPORTANT/flickr/photo_descriptions/'
    path_tweets = '/Users/kiki/Desktop/casia_cross_osn_local_data_IMPORTANT/twitter/tweets/'
    path_tweet_pics = '/Users/kiki/Desktop/casia_cross_osn_local_data_IMPORTANT/twitter/pred_tweet_pics_cross_osn/'
    model_names = ['bert-base-nli-mean-tokens', 'distilbert-base-nli-stsb-mean-tokens']
    model = SentenceTransformer(model_names[0])  # 0 or 1
    fl_keywords = True  # decide which data to consider (if both fl options are false fl_descriptions are utilized)
    fl_descriptions = True  # tweet text is always used, additionally tweet keywords can be considered
    tweet_keywords = True

    sbert_posts(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript, path_tweets,
                path_tweet_pics, model)

    """ bert-base-nli-mean-tokens
    with all data
    sbert + cosine similarity - Dataset A
    3 / 57
    0.05263157894736842

    without flickr keywords
    sbert + cosine similarity - Dataset A
    3 / 57
    0.05263157894736842

    without flickr description
    sbert + cosine similarity - Dataset A
    2 / 58
    0.034482758620689655

    without twitter keywords
    sbert + cosine similarity - Dataset A
    8 / 106
    0.07547169811320754

    without keywords
    sbert + cosine similarity - Dataset A
    8 / 108
    0.07407407407407407

    without fl_descriptions and without twitter keywords
    sbert + cosine similarity - Dataset A
    4 / 107
    0.037383177570093455
    """

    """ distilbert-base-nli-stsb-mean-tokens
        with all data
    sbert + cosine similarity - Dataset A
    3 / 57
    0.05263157894736842

    without flickr keywords
    sbert + cosine similarity - Dataset A
    3 / 57
    0.05263157894736842

    without flickr description
    sbert + cosine similarity - Dataset A
    1 / 58
    0.017241379310344827

    without twitter keywords
    sbert + cosine similarity - Dataset A
    12 / 106
    0.11320754716981132

    without keywords
    sbert + cosine similarity - Dataset A
    12 / 108
    0.1111111111111111

    without fl_descriptions and without twitter keywords
    sbert + cosine similarity - Dataset A
    4 / 107
    0.037383177570093455
    """