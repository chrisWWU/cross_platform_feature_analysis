import pandas as pd


def create_connect_realname(path_flickr_info, path_twitter_info, path_connect, path_to):
    """takes flickr info, twitter info, connect and returns df with pairs where both accounts have bio"""

    fl_info = pd.read_csv(path_flickr_info, index_col=0)
    tw_info = pd.read_csv(path_twitter_info, index_col=0, lineterminator='\n')

    connect = pd.read_csv(path_connect, index_col=0)

    # organize connect df (fl info, tw info, and connect need to be in same order)
    res = connect.loc[:, ['twitterusername', 'flickrid', 'flickrusername']]
    res.sort_values('twitterusername', inplace=True)  # sort according to twitter username
    res.reset_index(drop=True, inplace=True)

    # only keep info from users that are in res
    tw_info_new = tw_info[tw_info['username'].isin(res['twitterusername'])]

    # sort tw_info by twitter username
    tw_info_new.sort_values('username', inplace=True)
    tw_info_new.reset_index(drop=True, inplace=True)

    # sort connect df by twitter username
    connect.sort_values('twitterusername', inplace=True)
    connect.reset_index(drop=True, inplace=True)

    # sort fl_info's flickr id in the same order the flickr id's from connect df are sorted
    fl_info = fl_info.set_index('nsid')
    fl_info = fl_info.reindex(index=connect['flickrid'])
    fl_info.reset_index(drop=True, inplace=True)

    # get boolean for each row if twitter realname available
    res['twitter_realname_bool'] = tw_info_new['name'].notna()

    # get actual twitter realname
    res['twitter_realname'] = tw_info_new['name']

    # get boolean for each row if flickr realname available
    res['flickr_realname_bool'] = fl_info['realname'].notna()

    # get actual flickr realname
    res['flickr_realname'] = fl_info['realname']

    # only keep rows where twitter realname notna
    res = res.loc[res['twitter_realname_bool'], :].reset_index(drop=True)

    # only keep rows where flickr realname notna
    res = res.loc[res['flickr_realname_bool'], :].reset_index(drop=True)

    # drop bools
    res.drop(['twitter_realname_bool', 'flickr_realname_bool'], axis=1, inplace=True)

    res = res.reset_index(drop=True)

    # res now contains pairs of realnames
    if csv:
        res.to_csv(path_to)


if __name__ == '__main__':
    dataset = 'dataset_a'
    path_flickr_info = f'../../../../data/{dataset}/flickr/flickr_info.csv'
    path_twitter_info = f'../../../../data/{dataset}/twitter/twitter_info.csv'
    path_connect = f'../../../../data/{dataset}/connection.csv'
    path_to = 'realname_connection_personality.csv'
    csv = False

    create_connect_realname(path_flickr_info, path_twitter_info, path_connect, path_to)
