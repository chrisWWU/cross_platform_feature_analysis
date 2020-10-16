import pandas as pd
import numpy as np


def get_txt(path_from, path_to, path_res, csv):
    """transform edges.csv to .txt containing only integers as ids, also generate lookup table: int -> nsid"""
    df = pd.read_csv(path_from)
    ls = pd.Series()
    ls = ls.append(df['source'])
    ls = ls.append(df['target'])
    ls.reset_index(drop=True, inplace=True)

    r = dict(enumerate(set(ls)))
    r = {v: k for k, v in r.items()}

    df['source'] = df['source'].map(r)
    df['target'] = df['target'].map(r)

    lookup = pd.DataFrame(r.items())

    if csv:
        lookup.to_csv(path_res)
        np.savetxt(path_to, df.values, fmt='%d')


if __name__ == '__main__':
    # twitter core
    path_from = '../gephi_lists/twitter/twitter_core_edgelist.csv'
    path_to = 'twitter_core.txt'
    path_res = 'twitter_core_lookup.csv'
    csv = True

    get_txt(path_from, path_to, path_res, csv)

    # twitter total
    #path_from = '/Users/kiki/sciebo/gephi_graph_lists/cross_osn/twitter/twitter_edgelist.csv'
    #path_to = 'twitter.txt'
    #path_res = 'twitter_lookup.csv'
    #csv = True

    #get_txt(path_from, path_to, path_res, csv)
