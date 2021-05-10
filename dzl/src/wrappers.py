import inspect
from collections import defaultdict
from functools import reduce
from typing import Optional

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
                 cv_cls=StratifiedKFold,
                 n_folds: int = 5,
                 cv_params: Optional[dict] = None,
                 seeds: Optional[list] = None,
                 task: str = 'classification',
                 callbacks: Optional[list] = None,
                 *args, **kwargs):
        self.task: str = task
        assert self.task in self.tasks
        self.n_folds: int = n_folds
        self.cv_params: dict = cv_params
        self.seeds: list = seeds or [42]
        self.model_params: dict = model_params
        self.model_cls = model_cls
        self.cv_cls = cv_cls
        self.fold_models: defaultdict = defaultdict(dict)

        if self.task == 'classification':
            self._classes = None

        self.__fitted = False
        self.callbacks = []

    # utils
    def get_cv_obj(self, seed):
        cv_params = self.cv_params or dict(shuffle=True)
        cv_params['n_splits'] = self.n_folds
        cv_params['random_state'] = seed
        return self.cv_cls(**cv_params)

    def generate_folds(self, X, y):
        for seed in self.seeds:
            cv_obj = self.get_cv_obj(seed)
            for fold_id, (trn_idx, val_idx) in enumerate(cv_obj.split(X, y)):
                x_trn, x_val = X.iloc[trn_idx, :], X.iloc[val_idx, :]
                y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
                yield seed, fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val

    # any task
    def _fit(self, fold_model, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        for callback in self.callbacks:
            fold_model, x_trn, y_trn, x_val, y_val = callback.on_before_fit(self, fold_model, x_trn, y_trn, x_val,
                                                                            y_val,
                                                                            *args,
                                                                            **kwargs)
        if 'eval_set' in inspect.getfullargspec(fold_model.fit).args:
            fold_model.fit(x_trn, y_trn,
                           eval_set=[(x_val, y_val)],
                           *args, **kwargs)
        else:
            fold_model.fit(x_trn, y_trn, *args, **kwargs)
        for callback in self.callbacks:
            callback.on_after_fit(self, fold_model, x_trn, y_trn, x_val, y_val, *args, **kwargs)
        return self

    def fit(self, X, y, *args, **kwargs):
        if self.task == 'classification':
            self._classes = np.unique(y)
        for seed, fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
            fold_model = self.model_cls(**self.model_params)
            self.fold_models[seed][fold_id] = fold_model
            self._fit(fold_model, x_trn, y_trn, x_val, y_val, *args, **kwargs)
        return self

    # classification
    def _predict_proba(self, fold_model, X, *args, **kwargs):
        return fold_model.predict_proba(X, *args, **kwargs)

    # def _predict_proba_trn(self, X, y, *args, **kwargs):
    #     assert self.task == 'classification'
    #     oof = np.zeros(shape=(X.index.size, len(self._classes)))
    #     for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
    #         oof[trn_idx, :] += self._predict_proba(self.fold_models[fold_id], x_trn, *args, **kwargs) / (
    #                 self.cv_obj.n_splits - 1)
    #     return oof
    #
    # def _predict_proba_val(self, X, y, *args, **kwargs):
    #     assert self.task == 'classification'
    #     oof = np.zeros(shape=(X.index.size, len(self._classes)))
    #     for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
    #         oof[val_idx, :] = self._predict_proba(self.fold_models[fold_id], x_val, *args, **kwargs)
    #     return oof

    def predict_proba(self, X, *args, **kwargs):
        assert self.task == 'classification'
        oof = np.zeros(shape=(X.index.size, len(self._classes)))
        for fold_model in self.fold_models:
            oof += self._predict_proba(fold_model, X, *args, **kwargs) / (len(self.fold_models))
        return oof

    # classification
    def _predict(self, fold_model, X, *args, **kwargs):
        x_trn = X.copy()
        for callback in self.callbacks:
            x_trn = callback.on_before_predict(self, fold_model, X, x_trn, *args, **kwargs)

        results = fold_model.predict(x_trn, *args, **kwargs)

        for callback in self.callbacks:
            results = callback.on_after_predict(self, fold_model, X, x_trn, results, *args, **kwargs)

        return results

    # def _predict_trn(self, X, y, *args, **kwargs):
    #     oof = np.zeros(shape=X.index.size)
    #     for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
    #         oof[trn_idx] += self._predict(self.fold_models[fold_id], x_trn, *args, **kwargs) / (
    #                 self.cv_obj.n_splits - 1)
    #     return oof
    #
    # def _predict_val(self, X, y, *args, **kwargs):
    #     oof = np.zeros(shape=X.index.size)
    #     for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
    #         oof[val_idx] = self._predict(self.fold_models[fold_id], x_val, *args, **kwargs)
    #     return oof

    def fold_models_flatten(self):
        return reduce(lambda x, y: x + y, [list(d.values()) for d in self.fold_models.values()])

    def predict(self, X, *args, **kwargs):
        oof = np.zeros(shape=X.index.size)
        lst_models = self.fold_models_flatten()
        for fold_model in lst_models:
            oof += self._predict(fold_model, X, *args, **kwargs) / (len(lst_models))
        return oof


class BaseCallback:
    def on_before_fit(self, model, fold_model, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        return fold_model, x_trn, y_trn, x_val, y_val

    def on_after_fit(self, model, fold_model, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        return

    def on_before_predict(self, model, fold_model, X, x_trn, *args, **kwargs):
        return x_trn

    def on_after_predict(self, model, fold_model, X, x_trn, results, *args, **kwargs):
        return results


class ModelClassifierCV(BaseCVWrapper):
    pass
