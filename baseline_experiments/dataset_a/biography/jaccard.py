import pandas as pd
import numpy as np


def simJaccard(str1, str2):
    """calculate jaccard similarity"""
    str1 = set(str1.split())
    str2 = set(str2.split())
    if len(str1 | str2) == 0:
        return 0
    else:
        return float(len(str1 & str2)) / len(str1 | str2)


def jaccard_bio(path_connection):
    """calculate matching using jaccard similarity"""

    # read data
    df = pd.read_csv(path_connection, index_col=0, lineterminator='\n')

    # extract usernames
    fl_names = df['flickrusername'].tolist()
    tw_names = df['twitterusername'].tolist()

    # extract bios
    fl_bio = df['flickr_bio'].tolist()
    tw_bio = df['twitter_bio'].tolist()

    # initialize result
    result = pd.DataFrame(columns=['flickrusername', 'twitterusername'])
    bios = pd.DataFrame(columns=['flickrbio', 'twitterbio'])
    c = 0
    for i in fl_bio:

        # get similarity for each combination
        res = [simJaccard(i, j) for j in tw_bio]

        # select flickr user from current iteration
        fl_match = fl_names[c]

        # select corresponding twitter user with max similarity
        index_max = np.argmax(res)
        match = tw_names[index_max]

        # append pair to result
        result.loc[c] = [fl_match, match]

        # just to show bio matches:
        bios.loc[c] = [i, tw_bio[index_max]]

        c += 1

    #print(bios)

    # compare actual matching with jaccard matching
    df1 = result.merge(df, on=['flickrusername', 'twitterusername'], how='left', indicator='Exist')
    df1['Exist'] = np.where(df1.Exist == 'both', True, False)

    # count correct matches
    cor = df1['Exist'].sum()

    # print performance measures
    print('Jaccard - Dataset A')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':
    path_connection = 'bio_connection_personality.csv'

    jaccard_bio(path_connection)

# Jaccard & Levenstein Both 10 / 70
# correct: 10 / 70
# percentage: 0.14285714285714285

"""
Jaccard - Dataset A
10 / 65
0.15384615384615385


Jaccard - Dataset A
11 / 65
0.16923076923076924
"""