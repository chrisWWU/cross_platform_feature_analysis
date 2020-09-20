import pandas as pd
from clean_bio import clean_bios


def create_connect_bio(path_flickr_info, path_twitter_info, path_connect, path_to):
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

    # get boolean for each row if twitter bio available
    res['twitter_bio_bool'] = tw_info_new['bio'].notna()

    # get actual twitter bio
    res['twitter_bio'] = tw_info_new['bio']

    # get boolean for each row if flickr bio available
    res['flickr_bio_bool'] = fl_info['description'].notna()

    # get actual flickr bio
    res['flickr_bio'] = fl_info['description']

    # only keep rows where twitter bio notna
    res = res.loc[res['twitter_bio_bool'], :].reset_index(drop=True)

    # only keep rows where flickr bio notna
    res = res.loc[res['flickr_bio_bool'], :].reset_index(drop=True)

    # drop bools
    res.drop(['twitter_bio_bool', 'flickr_bio_bool'], axis=1, inplace=True)

    # extract bios
    fl_bio = res.pop('flickr_bio')
    tw_bio = res.pop('twitter_bio')

    # clean bios
    fl_bio = clean_bios(fl_bio)
    tw_bio = clean_bios(tw_bio)

    # get indices of empty bios after cleaning
    ind = [i for i, x in enumerate(fl_bio) if not x]
    ind.extend([i for i, x in enumerate(tw_bio) if not x])
    ind = sorted(set(ind))

    # remove users with empty bios after cleaning
    fl_bio = [i for j, i in enumerate(fl_bio) if j not in ind]
    tw_bio = [i for j, i in enumerate(tw_bio) if j not in ind]
    res = res.drop(res.index[ind])

    # add bios to result
    res['flickr_bio'] = fl_bio
    res['twitter_bio'] = tw_bio

    res = res.reset_index(drop=True)

    # res now contains pairs of clean bios
    if csv:
        res.to_csv(path_to)


if __name__ == '__main__':
    path_flickr_info = '/Users/kiki/sciebo/cross_osn/2_check_active/7_combined_flickr_info.csv'
    path_twitter_info = '/Users/kiki/sciebo/cross_osn/3_collect_data/twitter/1_twitter_bio_info.csv'
    path_connect = '/Users/kiki/sciebo/cross_osn/2_check_active/7_combined_connection.csv'
    path_to = 'bio_connection_cross_osn.csv'
    csv = True

    create_connect_bio(path_flickr_info, path_twitter_info, path_connect, path_to)