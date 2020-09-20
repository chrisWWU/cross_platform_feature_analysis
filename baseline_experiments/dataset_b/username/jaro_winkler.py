import pandas as pd
import numpy as np
import jellyfish


def jw_username(path_connection):
    """calculate matching using jaro winkler similarity"""

    # read data
    df = pd.read_csv(path_connection, index_col=0, lineterminator='\n')

    # extract usernames
    fl_names = df['flickrusername'].tolist()
    tw_names = df['twitterusername'].tolist()

    # initialize result
    result = pd.DataFrame(columns=['flickrusername', 'twitterusername'])
    c = 0
    for i in fl_names:

        # get similarity for each combination
        res = [jellyfish.jaro_winkler_similarity(i, j) for j in tw_names]

        # select flickr user from current iteration
        fl_match = fl_names[c]

        # select corresponding twitter user with max similarity
        index_max = np.argmax(res)
        match = tw_names[index_max]

        # append pair to result
        result.loc[c] = [fl_match, match]

        c += 1

    # compare actual matching with jaccard matching
    df1 = result.merge(df, on=['flickrusername', 'twitterusername'], how='left', indicator='Exist')
    df1['Exist'] = np.where(df1.Exist == 'both', True, False)

    # count correct matches
    cor = df1['Exist'].sum()

    # print performance measures
    print('Jaro Winkler- Dataset B')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':
    path_connection = '/Users/kiki/Desktop/casia_cross_osn_local_data_IMPORTANT/7_combined_connection.csv'

    jw_username(path_connection)

    """
    Jaro Winkler- Dataset B
    3645 / 5924
    0.6152937204591492
    """