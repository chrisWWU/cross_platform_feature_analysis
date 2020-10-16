from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import ParameterGrid
from ml_experiments.feature_reduction import decorrelation, del_base_features
from ml_experiments.plot_coefficients import f_importances, f_importances2
from ml_experiments.ml_execute.combined_datasets_small.flickr_as_training.merge_data import merge
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
    print(clf1.score(X_test, y_test))

    print('log loss:')
    print(log_loss(y_test, y_proba))

    threshold = 0.01
    y_proba[y_proba < threshold] = 0.0  # 0.05

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
    user = False
    real = False
    loc = False
    bio = False
    post = False
    pic = False
    friend = True

    svc, n = execute_ml(user, real, loc, bio, post, pic, friend)

"""
user real loc
log loss:
2.7669945200675197
0.01
980 / 987
0.9929078014184397

user real loc bio
log loss:
3.584236253496273
0.01
852 / 857
0.9941656942823804

user real loc friend
log loss:
3.025259989856315
0.01
961 / 971
0.9897013388259527
"""


