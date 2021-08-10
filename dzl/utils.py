from enum import Enum


class Task(Enum):
    Classification = 0
    Regression = 1


def generate_folds(cv_obj, X, y, sample_weight):
    for fold_id, (trn_idx, val_idx) in enumerate(cv_obj.split(X, y)):
        x_trn, x_val = X.iloc[trn_idx, :], X.iloc[val_idx, :]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        w_trn, w_val = sample_weight.iloc[trn_idx], sample_weight.iloc[val_idx]
        yield fold_id, trn_idx, val_idx, x_trn, y_trn, w_trn, x_val, y_val, w_val
