import pandas as pd
from PIL import Image
import imagehash
import numpy as np


def dhash(path_connection):
    """calculates profilepic similarity through image hashing + hamming distance"""
    connect = pd.read_csv(path_connection, index_col=0)
    result = pd.DataFrame(columns=['flickrid', 'twitterusername'])

    fl_paths = connect['fl_path']
    tw_paths = connect['tw_path']

    fl_id = connect['flickrid']
    tw_name = connect['twitterusername']

    for i in range(len(fl_paths)):
        print(f'{i + 1} / {len(fl_paths)}')

        res = [imagehash.dhash(Image.open(fl_paths[i])) - imagehash.dhash(Image.open(j)) for j in tw_paths]

        index_min = np.argmin(res)
        match = tw_name[index_min]
        result.loc[i] = [fl_id[i], match]

    df1 = result.merge(connect, on=['flickrid', 'twitterusername'], how='left', indicator='Exist')
    df1['Exist'] = np.where(df1.Exist == 'both', True, False)
    cor = df1['Exist'].sum()

    # uncomment to show matches
    #match = df1[df1['Exist'] == True].reset_index(drop=True)
    #match = match[['flickrid', 'twitterusername']]
    #match.to_csv('personality_profilepic_matches.csv')

    print('dHash - Dataset A')
    print(f'{cor} / {len(df1)}')  # correct / total
    print(cor / len(df1))  # accuracy


if __name__ == '__main__':
    path_connection = 'profilepic_connection_personality.csv'

    dhash(path_connection)

    """
    dHash - Dataset A
    18 / 108
    0.16666666666666666
    """