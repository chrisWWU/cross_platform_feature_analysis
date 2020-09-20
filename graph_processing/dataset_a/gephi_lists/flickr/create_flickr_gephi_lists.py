import pandas as pd
import os


def clear_filename(filename):
    return filename.replace('.csv', '')


def get_fl_nodelist(path_fl, path_connection, path_fl_nodelist, path_fl_core_nodelist, csv):
    """
    creates complete nodelist with labels 'core' and 'follow' readable by Gephi
    creates core nodelist containing flickrids usernames labels etc.

    """
    nsids = pd.Series()
    core_ids = []

    # iterate all .csv 'following' files, each file belongs to one core user
    for filename in os.listdir(path_fl):

        # append id to core user list
        core_ids.append(clear_filename(filename))

        # also append core user to complete ids
        nsids = nsids.append(pd.Series(clear_filename(filename)))

        # read following info
        df = pd.read_csv(path_fl + filename, index_col=0)

        if not df.empty:
            # append friend (following) contacts to complete id series
            nsids = nsids.append(df['nsid'])

    nsids = nsids.unique()

    # create nodelist
    nodelist = pd.DataFrame(columns=['id', 'label', 'timeset', 'relevant'])

    # fill complete ids
    nodelist['id'] = nsids

    # read connection info
    connect = pd.read_csv(path_connection, index_col=0).drop(['twitterid'], axis=1)

    # rename flickrid for merge
    connect.rename(columns={'flickrid': 'id'}, inplace=True)

    # label complete list as core or follow node
    nodelist.loc[nodelist['id'].isin(core_ids), 'relevant'] = 'core'
    nodelist['relevant'].fillna('follow', inplace=True)
    nodelist['label'] = nodelist['relevant']

    # create core nodelist by merging complete nodelist with connection df
    core_nodelist = pd.merge(nodelist, connect, on='id')
    core_nodelist['label'] = core_nodelist['flickrusername']

    if csv:
        #nodelist.to_csv(path_fl_nodelist, index=False)
        core_nodelist.to_csv(path_fl_core_nodelist, index=False)


def get_fl_edgelist(path_fl, path_fl_edgelist, path_fl_core_edgelist, csv):
    """
    creates complete edgelist
    creates core edgelist
    """
    core_ids = []
    edge_list = pd.DataFrame(columns=['source', 'target'])

    # iterate through all twitter follow files
    for filename in os.listdir(path_fl):

        # each file name is a core node name
        core_ids.append(clear_filename(filename))

        df = pd.read_csv(path_fl + filename, index_col=0)
        if not df.empty:

            # name of file is source node
            source_id = pd.Series(clear_filename(filename))

            # create df containing all edges of respective file
            inter_edge_list = pd.DataFrame(columns=['source', 'target'])

            # repeat source node to length of df
            inter_edge_list['source'] = source_id.repeat(len(df)).reset_index(drop=True)

            # add content of df as target column
            inter_edge_list['target'] = df['nsid']

            edge_list = edge_list.append(inter_edge_list)

    # create core edgelist by selecting all rows where target node is a core node
    core_edgelist = edge_list[edge_list['target'].isin(core_ids)]

    if csv:
        #edge_list.to_csv(path_fl_edgelist, index=False)
        core_edgelist.to_csv(path_fl_core_edgelist, index=False)


if __name__ == '__main__':
    path_fl = '/Users/kiki/sciebo/personality_trait_paper/flickr_and_twitter/flickr/following_flickr/'
    path_fl_nodelist = 'flickr_nodelist.csv'
    path_fl_core_nodelist = 'flickr_core_nodelist.csv'
    path_fl_edgelist = 'flickr_edgelist.csv'
    path_fl_core_edgelist = 'flickr_core_edgelist.csv'
    path_connection = '/Users/kiki/sciebo/personality_trait_paper/flickr_and_twitter/flickr/csv_flickr/6_connection_flickr_dataset.csv'
    csv = True

    get_fl_edgelist(path_fl, path_fl_edgelist, path_fl_core_edgelist, csv)
    get_fl_nodelist(path_fl, path_connection, path_fl_nodelist, path_fl_core_nodelist, csv)
