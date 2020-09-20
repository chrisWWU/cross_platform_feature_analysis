import pandas as pd
import numpy as np
import jellyfish


def jw_location(path_connection):
    """calculate matching using jaro winkler similarity"""

    # read data
    df = pd.read_csv(path_connection, index_col=0, lineterminator='\n')

    # extract usernames
    fl_names = df['flickrusername'].tolist()
    tw_names = df['twitterusername'].tolist()

    # extract locations
    fl_location = df['flickr_location'].tolist()
    tw_location = df['twitter_location'].tolist()

    # initialize result
    result = pd.DataFrame(columns=['flickrusername', 'twitterusername'])
    locations = pd.DataFrame(columns=['flickr_location', 'twitter_location'])
    c = 0
    for i in fl_location:

        # get similarity for each combination
        res = [jellyfish.jaro_winkler_similarity(i, j) for j in tw_location]

        # select flickr user from current iteration
        fl_match = fl_names[c]

        # select corresponding twitter user with max similarity
        index_max = np.argmax(res)
        match = tw_names[index_max]

        # append pair to result
        result.loc[c] = [fl_match, match]

        # just to show location matches:
        locations.loc[c] = [i, tw_location[index_max]]

        c += 1

    #print(locations)

    # compare actual matching with jaccard matching
    df1 = result.merge(df, on=['flickrusername', 'twitterusername'], how='left', indicator='Exist')
    df1['Exist'] = np.where(df1.Exist == 'both', True, False)

    # count correct matches
    cor = df1['Exist'].sum()

    # print performance measures
    print('Jaro Winkler - Dataset A')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':
    path_connection = 'location_connection_personality.csv'

    jw_location(path_connection)

    """
    Jaro Winkler - Dataset A
    16 / 43
    0.37209302325581395
    """