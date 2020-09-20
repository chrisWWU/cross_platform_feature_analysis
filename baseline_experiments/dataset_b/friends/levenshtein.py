import pandas as pd
import os
import jellyfish
import numpy as np


def clear_csv(s):
    return s.replace('.csv', '')


def compare_friends(path_fl, path_tw, path_connect, threshold):
    """compare usernames of followers of each profile using string similarity and match accounts that have most follower matches"""

    # get list of paths for followers
    fl_lists = os.listdir(path_fl)
    tw_lists = os.listdir(path_tw)

    # get list of flickr ids / twitterusername
    fl_ids = [clear_csv(i) for i in fl_lists]
    tw_names = [clear_csv(i) for i in tw_lists]

    # read connect
    connect = pd.read_csv(path_connect, index_col=0)

    fl_names = []

    # initialize result
    result = pd.DataFrame(columns=['flickrusername', 'twitterusername'])

    # get respective list of flickrusernames
    for i in fl_ids:
        name = connect.loc[connect['flickrid'] == i, 'flickrusername'].reset_index(drop=True)
        fl_names.append(name[0])

    c = 0
    for fl_user in fl_lists:  # iterate through flickr users
        print(fl_user)
        print(f'{c+1} / {len(fl_lists)}')

        fl = pd.read_csv(path_fl + fl_user)
        fl = fl['flickrusername']
        list_of_matches = []

        for tw_user in tw_lists:  # iterate through twitter users
            tw = pd.read_csv(path_tw + tw_user)
            if tw.empty:
                list_of_matches.append(0)
            else:
                tw = tw['target']
                tw = [clear_csv(s) for s in tw]

                match = 0
                count = 0

                for i in fl:  # iterate through flickr user's friends
                    # iterate through twitter user's friends
                    res = [jellyfish.levenshtein_distance(i, j) for j in tw]

                    # select max similarity value
                    min = np.min(res)
                    count += 1

                    # compare to threshold, check if match
                    if min < threshold:
                        match += 1
                list_of_matches.append(match)

        # select twitter user with most matches by index
        ind = np.argmax(list_of_matches)
        print(f'{fl_names[c]} and {clear_csv(tw_lists[ind])} have {list_of_matches[ind]} matches')

        # append pair to result
        result.loc[c] = [fl_names[c], tw_names[ind]]

        c +=1

    df1 = result.merge(connect, on=['flickrusername', 'twitterusername'], how='left', indicator='Exist')
    df1['Exist'] = np.where(df1.Exist == 'both', True, False)
    cor = df1['Exist'].sum()

    print(f'Levenshtein - Dataset B (threshold = {threshold})')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':

    dataset = 'dataset_b'
    path_fl = f'../../../graph_processing/{dataset}/baseline/flickr/'
    path_tw = f'../../../graph_processing/{dataset}/baseline/twitter/'
    path_connect = '/Users/kiki/Desktop/casia_cross_osn_local_data_IMPORTANT/7_combined_connection.csv'
    threshold = 3

    compare_friends(path_fl, path_tw, path_connect, threshold)

"""
Levenshtein 1
364 / 3796
0.0958904109589041

Levenshtein 2
446 / 3796
0.11749209694415173

levenshtein 2
439 / 3796
0.11564805057955743

Levenshtein 3
446 / 3796
0.11749209694415173

464 / 3796 (local)
0.12223393045310854

levenshtein 4
407 / 3796
0.10721812434141201

Levenshtein 5
286 / 3796
0.07534246575342465
"""


