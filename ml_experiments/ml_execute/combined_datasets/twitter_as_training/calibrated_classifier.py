from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import ParameterGrid
from ml_experiments.feature_reduction import decorrelation, del_base_features
from ml_experiments.plot_coefficients import f_importances, f_importances2
from ml_experiments.ml_execute.combined_datasets.twitter_as_training.merge_data import merge
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from sklearn.metrics import log_loss


def execute_ml(user, real, loc, bio, post, pic, friend):
    X_train, X_test = merge(user, real, loc, bio, post, pic, friend)

    # delete base features
    # X_train, X_test = del_base_features(X_train, X_test, user, real, loc)

    X_train = X_train.sample(frac=1, random_state=1).reset_index(drop=True)
    X_test = X_test.sample(frac=1, random_state=1).reset_index(drop=True)

    y_train = X_train.pop('label')
    y_test = X_test.pop('label')

    n = len(X_train)
    print('number of observations: ' + str(n))

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LinearSVC(max_iter=5000, dual=True, random_state=0)
    clf.fit(X_train, y_train)
    svc_acc = clf.score(X_test, y_test)
    print('linear svc accuracy: ' + str(svc_acc))

    clf1 = CalibratedClassifierCV(clf, cv="prefit")
    clf1.fit(X_train, y_train)

    y_proba = clf1.predict_proba(X_test)

    print('log loss:')
    print(log_loss(y_test, y_proba))

    threshold = 0.01
    y_proba[y_proba < threshold] = 0.0

    df = pd.DataFrame(y_proba, columns=clf1.classes_)

    # remove columns that only contain 0
    df = df.loc[:, (df != 0).any(axis=0)]

    # remove rows that only contain 0
    df = df.loc[(df != 0).any(1)]

    result = pd.DataFrame(columns=['actual', 'prediction'])
    c = 0
    for col in df.columns:
        ind = df[col].idxmax()
        pred = y_test[ind]
        result.loc[c] = [col, pred]
        c += 1

    C = np.where(result['actual'] == result['prediction'], 1, 0)

    print(threshold)
    print(f'{C.sum()} / {len(result)}')

    accuracy = C.sum() / len(result)
    print(accuracy)

    return svc_acc, n


if __name__ == '__main__':
    user = True
    real = True
    loc = True
    bio = True
    post = False
    pic = False
    friend = True

    svc, n = execute_ml(user, real, loc, bio, post, pic, friend)

"""
user real loc friend
log loss:
3.363343760593681
0.01
972 / 979
0.992849846782431

user real loc bio
log loss:
3.335401585170741
0.01
1304 / 1310
0.9954198473282443

user real loc
log loss:
2.9115804447857476
0.01
2147 / 2164
0.9921441774491682
"""


