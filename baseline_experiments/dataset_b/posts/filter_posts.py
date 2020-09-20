import pandas as pd
from os import listdir
from os.path import isfile, join
from clean_post import clean_posts
import pickle


def remove_csv(string):
    """deletes '.csv' from string"""
    return string.replace('.csv', '')


def remove_tweets(string):
    """deletes '_tweets.csv' from string"""
    return string.replace('_tweets.csv', '')


def match_posted_content(path_fl_pics, path_fl_descript, path_tweets, path_tweet_pics):
    """
    reads posted content and returnes 4 dicts of the form id/username -> content
    1. fl_pics_dict: flickrid -> image keywords
    2. fl_descript: flickrid -> image descriptions
    3. tweets_dict: twitter username -> tweet
    4. tweets_pic_dict: twitter username -> image keywords

    """

    # flickr pics ----------------------------------------------------
    fl_pics = [f for f in listdir(path_fl_pics) if isfile(join(path_fl_pics, f))]

    fl_pics_dict = {}
    for user in fl_pics:
        df = pd.read_csv(path_fl_pics + user, index_col=0)
        if not df.empty:
            # only use keywords with accuracy higher than 90%
            df = df[df['percentage'] > 90].reset_index(drop=True)
            if not df.empty:
                keywords = df['prediction'].tolist()
                keywords = [s.replace('_', ' ') for s in keywords]

                fl_pics_dict[remove_csv(user)] = keywords

    # flickr image descriptions --------------------------------------
    fl_descript = [f for f in listdir(path_fl_descript) if isfile(join(path_fl_descript, f))]

    fl_descript_dict = {}
    for user in fl_descript:
        df = pd.read_csv(path_fl_descript + user, index_col=0)
        if not df.empty:
            text = df[remove_csv(user)].tolist()
            text = [x for x in text if str(x) != 'nan']
            text = [str(x) for x in text]

            if text:
                # clean tweet text
                text = clean_posts(text)

                # remove space from empty strings
                text = [x.strip(' ') for x in text]

                # remove tweets that are empty after cleaning
                text = list(filter(None, text))

                # if not empty
                if text:
                    fl_descript_dict[remove_csv(user)] = text

    # tweets ----------------------------------------------------
    tweets = [f for f in listdir(path_tweets) if isfile(join(path_tweets, f))]

    tweets_dict = {}
    for user in tweets:
        df = pd.read_csv(path_tweets + user, lineterminator='\n', index_col=0)
        if not df.empty:
            text = df['tweet'].tolist()
            text = [x for x in text if str(x) != 'nan']
            text = [str(x) for x in text]

            # clean tweet text
            text = clean_posts(text)

            # remove space from empty strings
            text = [x.strip(' ') for x in text]

            # remove tweets that are empty after cleaning
            text = list(filter(None, text))

            # if not empty
            if text:
                tweets_dict[remove_tweets(user)] = text

    # tweet pics ----------------------------------------------------
    tweet_pics = [f for f in listdir(path_tweet_pics) if isfile(join(path_tweet_pics, f))]

    tweets_pic_dict = {}
    for user in tweet_pics:
        df = pd.read_csv(path_tweet_pics + user, index_col=0)
        if not df.empty:
            # only use keywords with accuracy higher than 90%
            df = df[df['percentage'] > 90].reset_index(drop=True)
            if not df.empty:
                keywords = df['prediction'].tolist()
                keywords = [s.replace('_', ' ') for s in keywords]

                tweets_pic_dict[remove_csv(user)] = keywords

    # save dict -----------------------------------------------------

    # filenames = [path_to_flickr_keywords, path_to_flickr_descript, path_to_tweets, path_to_tweet_keywords]
    # files = [fl_pics_dict, fl_descript_dict, tweets_dict, tweets_pic_dict]

    # for i in range(len(files)):
    #    with open(filenames[i], 'wb') as handle:
    #        pickle.dump(files[i], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return fl_pics_dict, fl_descript_dict, tweets_dict, tweets_pic_dict


def select_data(fl_pics_dict, fl_descript_dict, tweets_dict, tweets_pic_dict, path_connect, fl_keywords: bool,
                fl_descriptions: bool, tweet_keywords: bool):
    """
    takes in the 4 dicts and connection df and returns adjusted list of lists for flickr and twitter
    fl_keywords, fl_descriptions, tweet_keywords are boolean that determine whether to consider that data or not
    """
    connect = pd.read_csv(path_connect, index_col=0)

    # if both flickr types are false use image descriptions
    if fl_keywords == False and fl_descriptions == False:
        fl_descriptions = True

    if fl_keywords:
        # only keep rows where flickr keywords available
        connect = connect[connect['flickrid'].isin(fl_pics_dict.keys())].reset_index(drop=True)

    if fl_descriptions:
        # only keep rows where flickr descriptions available
        connect = connect[connect['flickrid'].isin(fl_descript_dict.keys())].reset_index(drop=True)

    # only keep rows where tweets are available
    connect = connect[connect['twitterusername'].isin(tweets_dict.keys())].reset_index(drop=True)

    if tweet_keywords:
        connect = connect[connect['twitterusername'].isin(tweets_pic_dict.keys())].reset_index(drop=True)

    # extract relevant ids / usernames
    nsids = connect['flickrid']
    twitterusernames = connect['twitterusername']

    # only keep relevant entries in flickr description dict
    if fl_descriptions:
        fl_description_list = []
        [fl_description_list.append(fl_descript_dict[id]) for id in nsids]
        print(len(fl_description_list))

    # only keep relevant entries in flickr keywords dict
    if fl_keywords:
        fl_keyword_list = []
        [fl_keyword_list.append(fl_pics_dict[id]) for id in nsids]
        print(len(fl_keyword_list))

    # only keep relevant entries in twitter keywords dict
    if tweet_keywords:
        tw_keyword_list = []
        [tw_keyword_list.append(tweets_pic_dict[id]) for id in twitterusernames]
        print(len(tw_keyword_list))

    # only keep relevant entries in tweets dict
    tweet_list = []
    [tweet_list.append(tweets_dict[id]) for id in twitterusernames]
    print(len(tweet_list))

    if tweet_keywords:
        # add twitter keywords to the respective tweet
        for i in range(len(tweet_list)):
            tweet_list[i].extend(tw_keyword_list[i])

    if fl_keywords and fl_descriptions:
        # add twitter keywords to the respective tweet
        for i in range(len(fl_description_list)):
            fl_description_list[i].extend(fl_keyword_list[i])

    if fl_descriptions:
        return fl_description_list, tweet_list, connect
    else:
        return fl_keyword_list, tweet_list, connect


def prepro(fl_keywords, fl_descriptions, tweet_keywords, path_connect, path_fl_pics, path_fl_descript, path_tweets,
           path_tweet_pics):

    a, b, c, d = match_posted_content(path_fl_pics, path_fl_descript, path_tweets, path_tweet_pics)

    fl, tw, connect = select_data(a, b, c, d, path_connect, fl_keywords, fl_descriptions, tweet_keywords)

    # convert flickr data to string
    fl = [' '.join(x) for x in fl]

    # convert twitter data to string
    tw = [' '.join(x) for x in tw]

    return fl, tw, connect
