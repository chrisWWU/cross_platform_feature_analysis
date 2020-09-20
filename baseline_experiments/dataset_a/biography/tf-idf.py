from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import util
import pandas as pd
import torch
import numpy as np
from nltk.stem.porter import *


def tfidf(path_connection):
    """calculate matching using tf-idf embeddings and cosine similarity"""

    # read data
    df = pd.read_csv(path_connection, index_col=0, lineterminator='\n')

    # extract usernames
    fl_names = df['flickrusername'].tolist()
    tw_names = df['twitterusername'].tolist()

    # extract bios
    fl_bio = df['flickr_bio'].tolist()
    tw_bio = df['twitter_bio'].tolist()

    # stem words
    ps = PorterStemmer()
    fl_bio = [ps.stem(x) for x in fl_bio]
    tw_bio = [ps.stem(x) for x in tw_bio]

    # Compute embedding for both lists
    vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase=True, stop_words='english', max_df=0.7, max_features=len(fl_bio))
    vector_fl = vectorizer.fit_transform(fl_bio)
    vector_tw = vectorizer.transform(tw_bio)

    # convert to tensor for faster cosine calculation
    fl_emb = torch.tensor(vector_fl.toarray())
    tw_emb = torch.tensor(vector_tw.toarray())

    # Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(fl_emb, tw_emb)
    cosine_scores[cosine_scores != cosine_scores] = 0

    # get indices of max similarity values
    ind = torch.argmax(cosine_scores, dim=1)

    # initialize result
    result = pd.DataFrame(columns=['flickrusername', 'twitterusername'])

    for i in range(len(fl_bio)):
        result.loc[i] = [fl_names[i], tw_names[ind[i]]]

    # compare actual matching with tfidf matching
    df1 = result.merge(df, on=['flickrusername', 'twitterusername'], how='left', indicator='Exist')
    df1['Exist'] = np.where(df1.Exist == 'both', True, False)

    # count correct matches
    cor = df1['Exist'].sum()

    # print performance measures
    print(f'tf-idf + cosine similarity - Dataset A')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':
    path_connection = 'bio_connection_personality.csv'

    tfidf(path_connection)

    """
    tf-idf + cosine similarity - Dataset A
    23 / 65
    0.35384615384615387
    
    with stemming:
    tf-idf + cosine similarity - Dataset A
    23 / 65
    0.35384615384615387
    
    with max_features = 1000
    tf-idf + cosine similarity - Dataset A
    21 / 65
    0.3230769230769231
    
    max_features = number of observations
    tf-idf + cosine similarity - Dataset A
    9 / 65
    0.13846153846153847
    """