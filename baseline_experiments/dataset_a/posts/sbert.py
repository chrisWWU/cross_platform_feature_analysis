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
    print(f'sbert + cosine similarity - Dataset A')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':
    path_connect = '/Users/kiki/sciebo/personality_trait_paper/flickr_and_twitter/flickr/csv_flickr/6_connection_flickr_dataset.csv'
    path_fl_pics = '/Users/kiki/sciebo/image_tagging/personality/flickr/flickr_pic_tags_personality/'
    path_fl_descript = '/Users/kiki/sciebo/personality_trait_paper/flickr_and_twitter/flickr/photo_descriptions/'
    path_tweets = '/Users/kiki/sciebo/personality_trait_paper/flickr_and_twitter/twitter_matching_flickr/csv_twitter/tweets/'
    path_tweet_pics = '/Users/kiki/sciebo/image_tagging/personality/twitter/pred_tweet_pics_personality/'
    model_names = ['bert-base-nli-mean-tokens', 'distilbert-base-nli-stsb-mean-tokens']
    model = SentenceTransformer(model_names[1])  # 0 or 1
    fl_keywords = True
    fl_descriptions = False
    tweet_keywords = False

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