import pandas as pd
import numpy as np
import jellyfish


def levenshtein_username(path_connection):
    """calculate matching using levenshtein distance"""

    # read data
    df = pd.read_csv(path_connection, index_col=0, lineterminator='\n')

    # make results reproducible
    df = df.sort_values(by='twitterusername').reset_index(drop=True)

    # extract usernames
    fl_names = df['flickrusername'].tolist()
    tw_names = df['twitterusername'].tolist()

    # initialize result
    result = pd.DataFrame(columns=['flickrusername', 'twitterusername'])
    c = 0
    for i in fl_names:

        # get similarity for each combination
        res = [jellyfish.levenshtein_distance(i, j) for j in tw_names]

        # select flickr user from current iteration
        fl_match = fl_names[c]

        # select corresponding twitter user with max similarity
        index_min = np.argmin(res)
        match = tw_names[index_min]

        # append pair to result
        result.loc[c] = [fl_match, match]

        c += 1

    # compare actual matching with jaccard matching
    df1 = result.merge(df, on=['flickrusername', 'twitterusername'], how='left', indicator='Exist')
    df1['Exist'] = np.where(df1.Exist == 'both', True, False)

    # count correct matches
    cor = df1['Exist'].sum()

    # print performance measures
    print('Levenshtein - Dataset B')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':
    dataset = 'dataset_b'
    path_connect = f'../../../../data/{dataset}/connection.csv'

    levenshtein_username(path_connect)

"""
Levenshtein - Dataset B
3619 / 5924
0.6109047940580689
"""