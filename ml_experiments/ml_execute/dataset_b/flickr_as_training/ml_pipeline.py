from sklearn import neighbors
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from ml_experiments.feature_reduction import decorrelation, del_base_features
from ml_experiments.plot_coefficients import f_importances
from ml_experiments.ml_execute.dataset_b.flickr_as_training.merge_data import merge


def execute_ml(user, real, loc, bio, post, pic, friend):

    X_train, X_test = merge(user, real, loc, bio, post, pic, friend)

    # delete base features
    #X_train, X_test = del_base_features(X_train, X_test, user, real, loc)


    X_train = X_train.sample(frac=1, random_state=0)
    X_test = X_test.sample(frac=1, random_state=0)

    y_train = X_train.pop('label')
    y_test = X_test.pop('label')

    n = len(X_train)
    print('number of observations: ' + str(n))

    feature_names = X_train.columns

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # decorrelation
    #X_train, X_test = decorrelation(X_train, X_test)


    clf = neighbors.KNeighborsClassifier(1, weights='uniform')
    clf.fit(X_train, y_train)
    knn_acc = clf.score(X_test, y_test)
    print('knn accuracy: ' + str(knn_acc))

    grid = {"C": [0.01, 0.1, 1, 10]}

    clf = LinearSVC(max_iter=5000, dual=True, random_state=0)
    best_score = 0
    best_grid = []
    for g in ParameterGrid(grid):
        print(g)
        clf.set_params(**g)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        # save if better than current best
        if clf.score(X_test, y_test) > best_score:
            best_score = score
            best_grid = g

    print(f"best svc score: {best_score}")
    print(f"best svc grid: {best_grid}")
    print('--------------------------------------------------------')

    #clf.fit(X_train, y_train)
    #svc_acc = clf.score(X_test, y_test)
    #print('linear svc accuracy: ' + str(svc_acc))

    # f_importances(abs(clf.coef_[0]), feature_names, top=30)

    return knn_acc, best_score, n


if __name__ == '__main__':

    user = True
    real = True
    loc = False
    bio = False
    post = False
    pic = False
    friend = False

    execute_ml(user, real, loc, bio, post, pic, friend)
