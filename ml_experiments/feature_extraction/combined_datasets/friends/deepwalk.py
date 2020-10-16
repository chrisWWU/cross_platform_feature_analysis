import networkx as nx
from graph_processing.ge import models
import numpy as np
import pandas as pd


def get_embeddings(path_connect, path_from, path_lookup, flickr):
    G = nx.read_edgelist(path_from, create_using=nx.DiGraph(), nodetype=str, data=[('weight', int)])  # Read graph

    model = models.DeepWalk(G, walk_length=10, num_walks=80, workers=1)  # init model
    model.train(window_size=5, iter=3)  # train model
    embeddings = model.get_embeddings()  # get embedding vectors

    res = np.zeros((len(embeddings), 128))

    for i in range(len(embeddings)):
        res[i, :] = embeddings[str(i)]

    res = pd.DataFrame(res)

    # change colnames to 'graph_' + number
    colnames = []
    for i in range(len(res.columns)):
        colnames.append(f'graph_{i}')
    res.columns = colnames

    # append label to embeddings (match with lookup table)
    ids = pd.DataFrame()
    lookup = pd.read_csv(path_lookup, index_col=0)

    # column to be matched depends whether its flickr or twitter data
    key = 'flickrid' if flickr else 'twitterusername'

    # read lookup
    ids[key] = lookup['0']

    connect = pd.read_csv(path_connect, index_col=0)

    # get matching usernames from connect table
    merged = pd.merge(ids, connect, on=key)

    fl_username = merged['flickrusername'].tolist()
    tw_username = merged['twitterusername'].tolist()

    # create label in format of 'flickrusername + twitterusername'
    labels = []
    for i in range(len(fl_username)):
        labels.append(f'{fl_username[i]} + {tw_username[i]}')

    # append label to embeddings
    res['label'] = pd.Series(labels)
    return res


if __name__ == '__main__':
    # flickr core
    dataset = 'combined_datasets'
    flickr = True
    path_connect = '/Users/kiki/sciebo/personality_trait_paper/flickr_and_twitter/flickr/csv_flickr/6_connection_flickr_dataset.csv'
    path_from = f'../../../../graph_processing/{dataset}/transformed_edgelists/flickr_core.txt'
    path_lookup = f'../../../../graph_processing/{dataset}/transformed_edgelists/flickr_core_lookup.csv'


    fl_emb = get_embeddings(path_connect, path_from, path_lookup, flickr)

    # twitter core
    dataset = 'combined_datasets'
    flickr = False
    path_connect = '/Users/kiki/sciebo/personality_trait_paper/flickr_and_twitter/flickr/csv_flickr/6_connection_flickr_dataset.csv'
    path_from = f'../../../../graph_processing/{dataset}/transformed_edgelists/twitter_core.txt'
    path_lookup = f'../../../../graph_processing/{dataset}/transformed_edgelists/twitter_core_lookup.csv'

    # -------------- only keep users present in both sets and write to csv ---------------------------------
    path_to_fl = f'../../../features/{dataset}/friends/deepwalk_flickr_core.csv'
    path_to_tw = f'../../../features/{dataset}/friends/deepwalk_twitter_core.csv'
    csv = False

    tw_emb = get_embeddings(path_connect, path_from, path_lookup, flickr)

    fl_emb = fl_emb[fl_emb['label'].isin(tw_emb['label'])].reset_index(drop=True)

    tw_emb = tw_emb[tw_emb['label'].isin(fl_emb['label'])].reset_index(drop=True)

    print('-------')
    print(len(fl_emb))
    print(len(tw_emb))

    if csv:
        fl_emb.to_csv(path_to_fl)
        tw_emb.to_csv(path_to_tw)
