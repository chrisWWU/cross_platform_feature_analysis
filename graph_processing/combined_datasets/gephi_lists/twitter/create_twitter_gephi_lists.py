import pandas as pd
import os


def clear_filename(filename):
    return filename.replace('.csv', '')


def get_tw_nodelist(path_tw, path_connection, path_tw_nodelist, path_tw_core_nodelist, csv):
    """
    creates complete nodelist with labels 'core' and 'follow' readable by Gephi
    creates core nodelist containing flickrids usernames labels etc.

    """
    names = pd.Series()
    core_names = []
    c = 0

    # iterate all .csv 'following' files, each file belongs to one core user
    for filename in os.listdir(path_tw):
        if (c+1) % 100 == 0:
            print(f'{c + 1} / {len(os.listdir(path_tw))}')
        c +=1

        # append id to core user list
        core_names.append(clear_filename(filename))

        # also append core user to complete ids
        names = names.append(pd.Series(clear_filename(filename)))

        # read following info
        df = pd.read_csv(path_tw + filename, index_col=0)

        if not df.empty:
            # append friend (following) contacts to complete id series
            names = names.append(df['0'])

    names = names.unique()

    # create nodelist
    nodelist = pd.DataFrame(columns=['id', 'label', 'timeset', 'relevant'])

    # fill complete names
    nodelist['id'] = names

    # read connection info
    connect = pd.read_csv(path_connection, index_col=0)

    # label complete list as core or follow node
    nodelist.loc[nodelist['id'].isin(core_names), 'relevant'] = 'core'
    nodelist['relevant'].fillna('follow', inplace=True)
    nodelist['label'] = nodelist['relevant']

    # rename twitterusername for merge
    connect.rename(columns={'twitterusername': 'id'}, inplace=True)

    # create core nodelist by merging complete nodelist with connection df
    core_nodelist = pd.merge(nodelist, connect, on='id')
    core_nodelist['label'] = core_nodelist['id']

    if csv:
        #nodelist.to_csv(path_tw_nodelist, index=False)
        core_nodelist.to_csv(path_tw_core_nodelist, index=False)


def get_tw_edgelist(path_tw, path_connection, path_tw_edgelist, path_tw_core_edgelist, csv):
    """
    creates complete edgelist
    creates core edgelist
    """

    # read connection info
    connect = pd.read_csv(path_connection, index_col=0)

    core_names = []
    edge_list = pd.DataFrame(columns=['source', 'target'])

    # only keep files that are in connect
    filenames = [x for x in os.listdir(path_tw) if clear_filename(x) in connect['twitterusername'].values]

    c = 0

    # iterate through all twitter follow files
    for filename in filenames:

        if (c+1) % 100 == 0:
            print(f'{c + 1} / {len(filenames)}')
        c +=1

        # each file name is a core node name
        core_names.append(clear_filename(filename))

        df = pd.read_csv(path_tw + filename, index_col=0)
        if not df.empty:

            # name of file is source node
            source_id = pd.Series(clear_filename(filename))

            # create df containing all edges of respective file
            inter_edge_list = pd.DataFrame(columns=['source', 'target'])

            # repeat source node to length of df
            inter_edge_list['source'] = source_id.repeat(len(df)).reset_index(drop=True)

            # add content of df as target column
            inter_edge_list['target'] = df['0']

            edge_list = edge_list.append(inter_edge_list)

    # create core edgelist by selecting all rows where target node is a core node
    core_edgelist = edge_list[edge_list['target'].isin(core_names)]

    if csv:
        #edge_list.to_csv(path_tw_edgelist, index=False)
        core_edgelist.to_csv(path_tw_core_edgelist, index=False)


if __name__ == '__main__':
    dataset = 'datasets_combined'

    path_tw = f'../../../../data/{dataset}/twitter/twitter_following/'
    path_tw_nodelist = 'twitter_nodelist.csv'
    path_tw_core_nodelist = 'twitter_core_nodelist.csv'
    path_tw_edgelist = 'twitter_edgelist.csv'
    path_tw_core_edgelist = 'twitter_core_edgelist.csv'
    path_connection = f'../../../../data/{dataset}/connection.csv'
    csv = False

    #get_tw_nodelist(path_tw, path_connection, path_tw_nodelist, path_tw_core_nodelist, csv)
    get_tw_edgelist(path_tw, path_connection, path_tw_edgelist, path_tw_core_edgelist, csv)


