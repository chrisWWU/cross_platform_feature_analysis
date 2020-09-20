import pandas as pd


def create_core_user_lists(path_fl, path_tw, path_to_fl, path_to_tw, path_connect):
    """create directory with csv for each core user containing his core user friends"""

    fl = pd.read_csv(path_fl)
    tw = pd.read_csv(path_tw)

    fl_total = fl['source'].append(fl['target'])
    tw_total = tw['source'].append(tw['target'])

    fl_unique = fl_total.unique()
    tw_unique = tw_total.unique()

    connect = pd.read_csv(path_connect, index_col=0)
    connect = connect[['flickrid', 'flickrusername']]

    for i in fl_unique:
        friends = fl.loc[fl['source'] == i, 'target'].reset_index(drop=True)
        df = pd.DataFrame(columns=['flickrid'])
        df['flickrid'] = friends
        df = df.merge(connect, on='flickrid', how='left')
        df.to_csv(path_to_fl + i + '.csv')

    for i in tw_unique:
        friends = tw.loc[tw['source'] == i, 'target'].reset_index(drop=True)
        friends.to_csv(path_to_tw + i + '.csv')


if __name__ == '__main__':

    # personality (dataset a)
    path_fl = '../gephi_lists/flickr/flickr_core_edgelist.csv'
    path_tw = '../gephi_lists/twitter/twitter_core_edgelist.csv'
    path_to_fl = f'flickr/'
    path_to_tw = f'twitter/'
    path_connect = '/Users/kiki/sciebo/personality_trait_paper/flickr_and_twitter/flickr/csv_flickr/6_connection_flickr_dataset.csv'

    create_core_user_lists(path_fl, path_tw, path_to_fl, path_to_tw, path_connect)