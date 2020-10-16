import itertools
import pandas as pd
from ml_experiments.ml_execute.dataset_b.flickr_as_training.ml_pipeline import execute_ml


def execute_all_combinations():
    """creates a grid with all combinations of 7 feature categories, calls Ml pipeline and saves results to csv"""

    path_to = 'results_flickr_minmax_base.csv'

    # create boolean df
    l = [False, True]
    res = [list(i) for i in itertools.product(l, repeat=7)]
    df = pd.DataFrame(res)

    # remove first row (where all entries are FALSE)
    df = df.drop([0]).reset_index(drop=True)

    # initialize result csv
    init_result = pd.DataFrame(columns=['user', 'real', 'loc', 'bio', 'post', 'pic', 'friend', 'number_users', 'knn', 'svc'])
    init_result.to_csv(path_to)

    # iterate through boolean df
    for i in range(len(df)):

        # get row from truth table
        params = [x for x in df.loc[i, :]]

        # call ML pipeline pass parameters
        knn_acc, svc_acc, n = execute_ml(*params)

        # append results to params
        params.extend([n, knn_acc, svc_acc])

        # open results csv
        result = pd.read_csv(path_to, index_col=0)

        # append new results
        result.loc[i] = params

        # write to csv
        result.to_csv(path_to)

if __name__ == '__main__':
    execute_all_combinations()
