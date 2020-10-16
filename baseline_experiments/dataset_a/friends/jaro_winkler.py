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

    # only keep users with friends
    fl_lists = [x for x in fl_lists if not pd.read_csv(path_fl + x).empty]
    tw_lists = [x for x in tw_lists if not pd.read_csv(path_tw + x).empty]

    # get list of flickr ids / twitterusername
    fl_ids = [clear_csv(i) for i in fl_lists]
    tw_names = [clear_csv(i) for i in tw_lists]

    # read connect
    connect = pd.read_csv(path_connect, index_col=0)

    # delete users without friends from connect df
    connect = connect[connect['flickrid'].isin(fl_ids)].reset_index(drop=True)
    connect = connect[connect['twitterusername'].isin(tw_names)].reset_index(drop=True)

    # make results reproducible
    connect = connect.sort_values(by='twitterusername').reset_index(drop=True)

    fl_ids = connect['flickrid'].tolist()
    tw_names = connect['twitterusername'].tolist()

    fl_lists = [x + '.csv' for x in fl_ids]
    tw_lists = [x + '.csv' for x in tw_names]

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
                    res = [jellyfish.jaro_winkler_similarity(i, j) for j in tw]

                    # select max similarity value
                    max = np.max(res)
                    count += 1

                    # compare to threshold, check if match
                    if max > threshold:
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

    print(f'Jaro Winkler - Dataset B (threshold = {threshold})')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':
    dataset = 'dataset_a'
    path_fl = f'../../../graph_processing/{dataset}/baseline/flickr/'
    path_tw = f'../../../graph_processing/{dataset}/baseline/twitter/'
    path_connect = f'../../../../data/{dataset}/connection.csv'
    threshold = 0.9  # 0.6 / 0.7 / 0.8 / 0.9

    compare_friends(path_fl, path_tw, path_connect, threshold)

"""
Jaro Winkler - Dataset A (threshold = 0.9)
3 / 4
0.75

"""





