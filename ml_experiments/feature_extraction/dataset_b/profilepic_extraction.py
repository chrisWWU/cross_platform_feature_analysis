import numpy as np
from PIL import Image
import os
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors


def clear_valid_path(r):
    return r.replace('.jpg', '')


def get_paths(path, connect, standard, colname):

    # create list of profile pic filenames
    pics = os.listdir(path)
    all_images_array = []

    # only use actual photos
    for each_file in pics:
        if each_file.endswith(".jpg") or each_file.endswith(".png"):
            # remove standard profilepic
            if open(path + each_file, 'rb').read() != open(standard, 'rb').read():
                all_images_array.append(each_file)

    valid_ids = [clear_valid_path(x) for x in all_images_array]

    # only user profilepics of users that are in connect
    connect = connect[connect[colname].isin(valid_ids)].reset_index(drop=True)

    # create path for each pic
    path_pics = [f'{path + pic}' for pic in all_images_array]
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


def merge(path_fl, path_tw, path_connect, fl_standard, tw_standard):

    connect = pd.read_csv(path_connect, index_col=0)

    fl_valid_path, fl_valid_ids, connect = get_paths(path_fl, connect, fl_standard, 'flickrid')
    tw_valid_path, tw_valid_ids, connect = get_paths(path_tw, connect, tw_standard, 'twitterusername')

    fl = pd.DataFrame()
    fl['flickrid'] = fl_valid_ids
    fl['fl_path'] = fl_valid_path

    tw = pd.DataFrame()
    tw['twitterusername'] = tw_valid_ids
    tw['tw_path'] = tw_valid_path

    connect = pd.merge(connect, fl, on='flickrid')
    connect = pd.merge(connect, tw, on='twitterusername')

    return connect


def image_operations(path):
    """takes profile pic path and returns list of features"""
    image = Image.open(path)
    image = image.convert('L')
    image = image.resize((10, 10))
    features = np.array(image).flatten()
    return list(features)


def extract_features(connect, path_to_fl, path_to_tw, csv):
    colnames = []
    for i in range(100):
        colnames.append(f'profilepic_{i}')

    fl_features = [image_operations(i) for i in connect['fl_path']]
    tw_features = [image_operations(i) for i in connect['tw_path']]

    X_fl = pd.DataFrame(fl_features, columns=colnames)
    X_tw = pd.DataFrame(tw_features, columns=colnames)

    fl_username = connect['flickrusername']
    tw_username = connect['twitterusername']

    res = []
    for i in range(len(fl_username)):
        res.append(f'{fl_username[i]} + {tw_username[i]}')

    X_fl['label'] = pd.Series(res)
    X_tw['label'] = pd.Series(res)

    if csv:
        X_fl.to_csv(path_to_fl)
        X_tw.to_csv(path_to_tw)

    X_fl = X_fl.sample(frac=1, random_state=0)
    X_tw = X_tw.sample(frac=1, random_state=0)

    y_fl = X_fl.pop('label')
    y_tw = X_tw.pop('label')

    scaler = StandardScaler(with_mean=False)
    X_fl = scaler.fit_transform(X_fl)
    X_tw = scaler.transform(X_tw)

    clf = neighbors.KNeighborsClassifier(1, weights='uniform')
    clf.fit(X_fl, y_fl)
    print(clf.score(X_tw, y_tw))

    clf = LinearSVC(max_iter=10000)
    clf.fit(X_fl, y_fl)
    print(clf.score(X_tw, y_tw))


if __name__ == '__main__':

    csv = False

    dataset = 'dataset_b'

    path_connect = f'../../../../data/{dataset}/connection.csv'

    path_fl = f'../../../../data/{dataset}/flickr/flickr_profilepics/'
    path_tw = f'../../../../data/{dataset}/twitter/twitter_profilepics/'

    fl_standard = f'../../../../data/{dataset}/flickr/flickr_profilepics/55578087@N08.jpg'
    tw_standard = f'../../../../data/{dataset}/twitter/twitter_profilepics/shardproducton.jpg'

    path_to_fl = '../../features/dataset_b/profilepic_fl.csv'
    path_to_tw = '../../features/dataset_b/profilepic_tw.csv'

    connect = merge(path_fl, path_tw, path_connect, fl_standard, tw_standard)
    extract_features(connect, path_to_fl, path_to_tw, csv)

    """
    knn: 0.07484639731893503
    svc: 0.07279836157140197
    """