import pandas as pd
import numpy as np
import jellyfish


def levenshtein_realname(path_connection):
    """calculate matching using levenshtein distance"""

    # read data
    df = pd.read_csv(path_connection, index_col=0, lineterminator='\n')

    # make results reproducible
    df = df.sort_values(by='twitterusername').reset_index(drop=True)

    # extract usernames
    fl_names = df['flickrusername'].tolist()
    tw_names = df['twitterusername'].tolist()

    # extract realnames
    fl_realname = df['flickr_realname'].tolist()
    tw_realname = df['twitter_realname'].tolist()

    # initialize result
    result = pd.DataFrame(columns=['flickrusername', 'twitterusername'])
    realnames = pd.DataFrame(columns=['flickr_realname', 'twitter_realname'])
    c = 0
    for i in fl_realname:

        # get similarity for each combination
        res = [jellyfish.levenshtein_distance(i, j) for j in tw_realname]

        # select flickr user from current iteration
        fl_match = fl_names[c]

        # select corresponding twitter user with max similarity
        index_min = np.argmin(res)
        match = tw_names[index_min]

        # append pair to result
        result.loc[c] = [fl_match, match]

        # just to show realname matches:
        realnames.loc[c] = [i, tw_realname[index_min]]

        c += 1

    #print(realnames)

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
    path_connection = 'realname_connection_cross_osn.csv'

    levenshtein_realname(path_connection)

    """
    Levenshtein - Dataset B
    3037 / 4465
    0.6801791713325868
    """