import pandas as pd


def create_connect_location(path_flickr_info, path_twitter_info, path_connect, path_to):
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

    # get boolean for each row if twitter location available
    res['twitter_location_bool'] = tw_info_new['location'].notna()

    # get actual twitter location
    res['twitter_location'] = tw_info_new['location']

    # get boolean for each row if flickr location available
    res['flickr_location_bool'] = fl_info['location'].notna()

    # get actual flickr location
    res['flickr_location'] = fl_info['location']

    # only keep rows where twitter location notna
    res = res.loc[res['twitter_location_bool'], :].reset_index(drop=True)

    # only keep rows where flickr location notna
    res = res.loc[res['flickr_location_bool'], :].reset_index(drop=True)

    # drop bools
    res.drop(['twitter_location_bool', 'flickr_location_bool'], axis=1, inplace=True)

    res = res.reset_index(drop=True)

    # res now contains pairs of locations
    if csv:
        res.to_csv(path_to)


if __name__ == '__main__':
    dataset = 'dataset_a'
    path_flickr_info = f'../../../../data/{dataset}/flickr/flickr_info.csv'
    path_twitter_info = f'../../../../data/{dataset}/twitter/twitter_info.csv'
    path_connect = f'../../../../data/{dataset}/connection.csv'

    path_to = 'location_connection_personality.csv'
    csv = False

    create_connect_location(path_flickr_info, path_twitter_info, path_connect, path_to)
