import inspect

import numpy as np
from sklearn.model_selection import StratifiedKFold


def generate_folds(cv_obj, X, y):
    for fold_id, (trn_idx, val_idx) in enumerate(cv_obj.split(X, y)):
        x_trn, x_val = X.iloc[trn_idx, :], X.iloc[val_idx, :]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        yield fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val


class BaseCVWrapper:
    tasks = ['regression', 'classification']

    def __init__(self,
                 model_cls,
                 model_params,
                 cv_obj=StratifiedKFold(n_splits=5, random_state=42, shuffle=True),
                 task='classification', *args, **kwargs):
        self.task = task
        assert self.task in self.tasks
        self.cv_obj = cv_obj
        self.model_params = model_params
        self.models = [model_cls(**self.model_params) for fold_id in range(self.cv_obj.n_splits)]
        if self.task == 'classification':
            self._classes = None

        self.__fitted = False

    # utils
    def generate_folds(self, X, y):
        for fold_id, (trn_idx, val_idx) in enumerate(self.cv_obj.split(X, y)):
            x_trn, x_val = X.iloc[trn_idx, :], X.iloc[val_idx, :]
            y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
            yield fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val

    # any task
    def _fit(self, model, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        if 'eval_set' in inspect.getfullargspec(model.fit).args:
            model.fit(x_trn, y_trn,
                      eval_set=[(x_val, y_val)],
                      *args, **kwargs)
        else:
            model.fit(x_trn, y_trn, *args, **kwargs)
        return self

    def fit(self, X, y, *args, **kwargs):
        if self.task == 'classification':
            self._classes = np.unique(y)
        for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
            self._fit(self.models[fold_id], x_trn, y_trn, x_val, y_val, *args, **kwargs)
        return self

    # classification
    def _predict_proba(self, model, X, *args, **kwargs):
        return model.predict_proba(X, *args, **kwargs)

    def predict_proba_trn(self, X, y, *args, **kwargs):
        assert self.task == 'classification'
        oof = np.zeros(shape=(X.index.size, len(self._classes)))
        for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
            oof[trn_idx, :] += self._predict_proba(self.models[fold_id], x_trn, *args, **kwargs) / (
                    self.cv_obj.n_splits - 1)
        return oof

    def predict_proba_val(self, X, y, *args, **kwargs):
        assert self.task == 'classification'
        oof = np.zeros(shape=(X.index.size, len(self._classes)))
        for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
            oof[val_idx, :] = self._predict_proba(self.models[fold_id], x_val, *args, **kwargs)
        return oof

    def predict_proba(self, X, *args, **kwargs):
        assert self.task == 'classification'
        oof = np.zeros(shape=(X.index.size, len(self._classes)))
        for model in self.models:
            oof += self._predict_proba(model, X, *args, **kwargs) / self.cv_obj.n_splits
        return oof

    # classification
    def _predict(self, model, X, *args, **kwargs):
        return model.predict(X, *args, **kwargs)

    def predict_trn(self, X, y, *args, **kwargs):
        oof = np.zeros(shape=X.index.size)
        for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
            oof[trn_idx, :] += self._predict(self.models[fold_id], x_trn, *args, **kwargs) / (
                    self.cv_obj.n_splits - 1)
        return oof

    def predict_val(self, X, y, *args, **kwargs):
        oof = np.zeros(shape=X.index.size)
        for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
            oof[val_idx, :] = self._predict(self.models[fold_id], x_val, *args, **kwargs)
        return oof

    def predict(self, X, *args, **kwargs):
        oof = np.zeros(shape=X.index.size)
        for model in self.models:
            oof += self._predict(model, X, *args, **kwargs) / self.cv_obj.n_splits
        return oof


class ModelClassifierCV(BaseCVWrapper):
    pass
