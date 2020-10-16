import pandas as pd

def merge(user, real, loc, bio, post, pic, friend):

    user_train = '../../../features/dataset_b/voc_learned_twitter/username_train.csv'
    user_test = '../../../features/dataset_b/voc_learned_twitter/username_test.csv'

    real_train = '../../../features/dataset_b/voc_learned_twitter/realname_train.csv'
    real_test = '../../../features/dataset_b/voc_learned_twitter/realname_test.csv'

    loc_train = '../../../features/dataset_b/voc_learned_twitter/location_train.csv'
    loc_test = '../../../features/dataset_b/voc_learned_twitter/location_test.csv'

    bio_train = '../../../features/dataset_b/voc_learned_twitter/bio_train.csv'
    bio_test = '../../../features/dataset_b/voc_learned_twitter/bio_test.csv'

    post_train = '../../../features/dataset_b/voc_learned_twitter/post_train.csv'
    post_test = '../../../features/dataset_b/voc_learned_twitter/post_test.csv'

    pic_train = '../../../features/dataset_b/profilepic_tw.csv'
    pic_test = '../../../features/dataset_b/profilepic_fl.csv'

    friend_train = '../../../features/dataset_b/friends/sdne_twitter_core.csv'
    friend_test = '../../../features/dataset_b/friends/sdne_flickr_core.csv'

    # create initial df with complete labels to merge
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()

    X_train['label'] = pd.read_csv(user_train, index_col=0)['label']
    X_test['label'] = pd.read_csv(user_test, index_col=0)['label']

    if user:
        X_train_user = pd.read_csv(user_train, index_col=0)
        X_test_user = pd.read_csv(user_test, index_col=0)
        print('username features: ' + str(len(X_train_user.columns)-1))

        X_train = pd.merge(X_train, X_train_user, on='label', how='inner')
        X_test = pd.merge(X_test, X_test_user, on='label', how='inner')

    if real:
        X_train_realname = pd.read_csv(real_train, index_col=0)
        X_test_realname = pd.read_csv(real_test, index_col=0)
        print('realname features: ' + str(len(X_train_realname.columns)-1))

        X_train = pd.merge(X_train, X_train_realname, on='label', how='inner')
        X_test = pd.merge(X_test, X_test_realname, on='label', how='inner')

    if loc:
        X_train_location = pd.read_csv(loc_train, index_col=0)
        X_test_location = pd.read_csv(loc_test, index_col=0)
        print('location features: ' + str(len(X_train_location.columns)-1))

        X_train = pd.merge(X_train, X_train_location, on='label', how='inner')
        X_test = pd.merge(X_test, X_test_location, on='label', how='inner')

    if bio:
        X_train_bio = pd.read_csv(bio_train, index_col=0)
        X_test_bio = pd.read_csv(bio_test, index_col=0)
        print('bio features: ' + str(len(X_train_bio.columns)-1))

        X_train = pd.merge(X_train, X_train_bio, on='label', how='inner')
        X_test = pd.merge(X_test, X_test_bio, on='label', how='inner')

    if post:
        X_train_post = pd.read_csv(post_train, index_col=0)
        X_test_post = pd.read_csv(post_test, index_col=0)
        print('post features: ' + str(len(X_train_post.columns)-1))

        X_train = pd.merge(X_train, X_train_post, on='label', how='inner')
        X_test = pd.merge(X_test, X_test_post, on='label', how='inner')

    if pic:
        X_train_pic = pd.read_csv(pic_train, index_col=0)
        X_test_pic = pd.read_csv(pic_test, index_col=0)
        print('profilepic features: ' + str(len(X_train_pic.columns)-1))

        X_train = pd.merge(X_train, X_train_pic, on='label', how='inner')
        X_test = pd.merge(X_test, X_test_pic, on='label', how='inner')

    if friend:
        X_train_friend = pd.read_csv(friend_train, index_col=0)
        X_test_friend = pd.read_csv(friend_test, index_col=0)
        print('friend features: ' + str(len(X_train_friend.columns)-1))

        X_train = pd.merge(X_train, X_train_friend, on='label', how='inner')
        X_test = pd.merge(X_test, X_test_friend, on='label', how='inner')

    return X_train, X_test