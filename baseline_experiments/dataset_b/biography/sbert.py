from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import numpy as np


def bert_bio(path_connection, model):
    """calculate matching using Sentence Bert embeddings and cosine similarity"""

    # read data
    df = pd.read_csv(path_connection, index_col=0, lineterminator='\n')

    # extract usernames
    fl_names = df['flickrusername'].tolist()
    tw_names = df['twitterusername'].tolist()

    # extract bios
    fl_bio = df['flickr_bio'].tolist()
    tw_bio = df['twitter_bio'].tolist()

    # Compute embedding for both lists
    fl_sentence_embeddings = model.encode(fl_bio, convert_to_tensor=True)
    tw_sentence_embeddings = model.encode(tw_bio, convert_to_tensor=True)

    # Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(fl_sentence_embeddings, tw_sentence_embeddings)

    # get indices of max similarity values
    ind = torch.argmax(cosine_scores, dim=1)

    # initialize result
    result = pd.DataFrame(columns=['flickrusername', 'twitterusername'])

    for i in range(len(fl_bio)):
        result.loc[i] = [fl_names[i], tw_names[ind[i]]]

    # compare actual matching with sbert matching
    df1 = result.merge(df, on=['flickrusername', 'twitterusername'], how='left', indicator='Exist')
    df1['Exist'] = np.where(df1.Exist == 'both', True, False)

    # count correct matches
    cor = df1['Exist'].sum()

    # print performance measures
    print(f'SBERT + cosine similarity- Dataset B - Model: {model_name}')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':
    path_connection = 'bio_connection_cross_osn.csv'
    model_names = ['bert-base-nli-mean-tokens', 'distilbert-base-nli-stsb-mean-tokens']
    model = SentenceTransformer(model_names[1])  # 0 or 1

    bert_bio(path_connection, model)

    """    
    SBERT - Dataset B - Model: bert-base-nli-mean-tokens
    224 / 2880
    0.07777777777777778
    
    SBERT - Dataset B - Model: distilbert-base-nli-stsb-mean-tokens
    256 / 2880
    0.08888888888888889
    
    """