import pandas as pd
import os
from PIL import Image


def clear_valid_path(r):
    #r = path.split('/')[8]
    return r.replace('.jpg', '')


def get_flickr_paths(path_fl, connect, flickr_standard):

    # create list of profile pic filenames
    pics = os.listdir(path_fl)
    all_images_array = []

    # only use actual photos
    for each_file in pics:
        if each_file.endswith(".jpg") or each_file.endswith(".png"):
            # remove standard profilepic
            if open(path_fl + each_file, 'rb').read() != open(flickr_standard, 'rb').read():
                all_images_array.append(each_file)


    valid_ids = [clear_valid_path(x) for x in all_images_array]

    # only user profilepics of users that are in connect
    connect = connect[connect['flickrid'].isin(valid_ids)].reset_index(drop=True)


    # create path for each pic
    path_pics = [f'{path_fl + pic}' for pic in all_images_array]
    valid_paths = []

    # check for each image if its broken
    c = 0
    for path in path_pics:
        try:
            im = Image.open(path)
            valid_paths.append(path)

        except IOError:
            del valid_ids[c]
            print(f'{path}: image is broken')
        c += 1

    return valid_paths, valid_ids, connect


def get_twitter_paths(path_tw, connect, twitter_standard):

    # create list of profile pic filenames
    pics = os.listdir(path_tw)
    all_images_array = []

    # only use actual photos
    for each_file in pics:
        if each_file.endswith(".jpg") or each_file.endswith(".png"):
            # remove standard profilepic
            if open(path_tw + each_file, 'rb').read() != open(twitter_standard, 'rb').read():
                all_images_array.append(each_file)

    valid_ids = [clear_valid_path(x) for x in all_images_array]

    # only user profilepics of users that are in connect
    connect = connect[connect['twitterusername'].isin(valid_ids)].reset_index(drop=True)

    # create path for each pic
    path_pics = [f'{path_tw + pic}' for pic in all_images_array]
    valid_paths = []

    # check for each image if its broken
    c = 0
    for path in path_pics:
        try:
            im = Image.open(path)
            valid_paths.append(path)

        except IOError:
            del valid_ids[c]

            print(f'{path}: image is broken')
        c += 1

    return valid_paths, valid_ids, connect


def execute(path_fl, path_tw, path_connect, path_to, flickr_standard, twitter_standard, csv):

    connect = pd.read_csv(path_connect, index_col=0)

    fl_valid_path, fl_valid_ids, connect = get_flickr_paths(path_fl, connect, flickr_standard)

    tw_valid_path, tw_valid_ids, connect = get_twitter_paths(path_tw, connect, twitter_standard)

    fl = pd.DataFrame()
    fl['flickrid'] = fl_valid_ids
    fl['fl_path'] = fl_valid_path

    tw = pd.DataFrame()
    tw['twitterusername'] = tw_valid_ids
    tw['tw_path'] = tw_valid_path

    connect = pd.merge(connect, fl, on='flickrid')
    connect = pd.merge(connect, tw, on='twitterusername')

    if csv:
        connect.to_csv(path_to)


if __name__ == '__main__':
    path_connect = '/Users/kiki/sciebo/personality_trait_paper/flickr_and_twitter/flickr/csv_flickr/6_connection_flickr_dataset.csv'
    path_fl = '/Users/kiki/sciebo/personality_trait_paper/flickr_and_twitter/flickr/profile_pics_flickr/'
    path_tw = '/Users/kiki/sciebo/personality_trait_paper/flickr_and_twitter/twitter_matching_flickr/profile_pics_twitter/'
    path_to = 'profilepic_connection_personality.csv'

    flickr_standard = '/Users/kiki/sciebo/personality_trait_paper/flickr_and_twitter/flickr/profile_pics_flickr/55578087@N08.jpg'
    twitter_standard = '/Users/kiki/sciebo/personality_trait_paper/flickr_and_twitter/twitter_matching_flickr/profile_pics_twitter/shardproducton.jpg'
    csv = True

    execute(path_fl, path_tw, path_connect, path_to, flickr_standard, twitter_standard, csv)