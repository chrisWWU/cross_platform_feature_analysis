from nancorrmp.nancorrmp import NaNCorrMp
import numpy as np
import pandas as pd

import psutil
"""
import h5py
from joblib import Parallel, delayed
from scipy import stats
import sys
from tqdm import trange
"""

def del_base_features(X_train, X_test, user, real, loc):
    """remove base features from training and test data"""

    remove = []
    base_user = ['user_numbers', 'user_letters', 'user_spaces', 'user_other_chars', 'user_number_perc',
                 'user_letter_perc']
    base_real = ['real_numbers', 'real_letters', 'real_spaces', 'real_other_chars', 'real_number_perc',
                 'real_letter_perc']
    base_loc = ['loc_numbers', 'loc_letters', 'loc_spaces', 'loc_other_chars', 'loc_number_perc', 'loc_letter_perc']

    if user:
        remove.extend(base_user)

    if real:
        remove.extend(base_real)

    if loc:
        remove.extend(base_loc)

    before = len(X_train.columns)

    X_train = X_train.drop(remove, axis=1)
    X_test = X_test.drop(remove, axis=1)

    after = len(X_train.columns)
    print(f'number of base features removed: {before - after}')

    return X_train, X_test



def decorrelation(df_train, df_test):
    correlation_threshold = 0.9

    df_train = pd.DataFrame(df_train)
    df_test = pd.DataFrame(df_test)

    available_memory = psutil.virtual_memory().available
    required_memory_for_correlation_matrix = (df_train.shape[1] ** 2) * 8

    if available_memory > (2.0 * required_memory_for_correlation_matrix):  # factor of 2 for r and p matrices

        print("Running NaNCorrMp", flush=True)
        np.seterr(divide='ignore', invalid='ignore')
        df_r, df_p = NaNCorrMp.calculate_with_p_value(df_train)
        del df_p
        upper = df_r.where(np.triu(np.ones(df_r.shape), k=1).astype(np.bool))
        del df_r

        correlated_features = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        print("Number of correlated features to drop: {}".format(len(correlated_features)), flush=True)

        df_train.drop(correlated_features, axis=1, inplace=True)
        df_test.drop(correlated_features, axis=1, inplace=True)

    else:
        print('too large for decorrelation')

    return df_train, df_test


