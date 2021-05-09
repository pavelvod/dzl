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
        self.fold_models = [model_cls(**self.model_params) for fold_id in range(self.cv_obj.n_splits)]
        if self.task == 'classification':
            self._classes = None

        self.__fitted = False
        self.callbacks = []

    # utils
    def generate_folds(self, X, y):
        for fold_id, (trn_idx, val_idx) in enumerate(self.cv_obj.split(X, y)):
            x_trn, x_val = X.iloc[trn_idx, :], X.iloc[val_idx, :]
            y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
            yield fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val

    # any task
    def _fit(self, fold_model, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        for callback in self.callbacks:
            fold_model, x_trn, y_trn, x_val, y_val = callback.on_before_fit(fold_model, x_trn, y_trn, x_val, y_val,
                                                                            *args,
                                                                            **kwargs)
        if 'eval_set' in inspect.getfullargspec(fold_model.fit).args:
            fold_model.fit(x_trn, y_trn,
                           eval_set=[(x_val, y_val)],
                           *args, **kwargs)
        else:
            fold_model.fit(x_trn, y_trn, *args, **kwargs)
        for callback in self.callbacks:
            fold_model, x_trn, y_trn, x_val, y_val = callback.on_after_fit(fold_model, x_trn, y_trn, x_val, y_val,
                                                                           *args,
                                                                           **kwargs)
        return self

    def fit(self, X, y, *args, **kwargs):
        if self.task == 'classification':
            self._classes = np.unique(y)
        for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
            self._fit(self.fold_models[fold_id], x_trn, y_trn, x_val, y_val, *args, **kwargs)
        return self

    # classification
    def _predict_proba(self, fold_model, X, *args, **kwargs):
        return fold_model.predict_proba(X, *args, **kwargs)

    def _predict_proba_trn(self, X, y, *args, **kwargs):
        assert self.task == 'classification'
        oof = np.zeros(shape=(X.index.size, len(self._classes)))
        for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
            oof[trn_idx, :] += self._predict_proba(self.fold_models[fold_id], x_trn, *args, **kwargs) / (
                    self.cv_obj.n_splits - 1)
        return oof

    def _predict_proba_val(self, X, y, *args, **kwargs):
        assert self.task == 'classification'
        oof = np.zeros(shape=(X.index.size, len(self._classes)))
        for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
            oof[val_idx, :] = self._predict_proba(self.fold_models[fold_id], x_val, *args, **kwargs)
        return oof

    def predict_proba(self, X, *args, **kwargs):
        assert self.task == 'classification'
        oof = np.zeros(shape=(X.index.size, len(self._classes)))
        for fold_model in self.fold_models:
            oof += self._predict_proba(fold_model, X, *args, **kwargs) / self.cv_obj.n_splits
        return oof

    # classification
    def _predict(self, fold_model, X, *args, **kwargs):
        for callback in self.callbacks:
            fold_model, X = callback.on_before_predict(fold_model, X, *args, **kwargs)
        results = fold_model.predict(X, *args, **kwargs)

        for callback in self.callbacks:
            fold_model, X, results = callback.on_before_predict(fold_model, X, results, *args, **kwargs)

        return results

    def _predict_trn(self, X, y, *args, **kwargs):
        oof = np.zeros(shape=X.index.size)
        for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
            oof[trn_idx] += self._predict(self.fold_models[fold_id], x_trn, *args, **kwargs) / (
                    self.cv_obj.n_splits - 1)
        return oof

    def _predict_val(self, X, y, *args, **kwargs):
        oof = np.zeros(shape=X.index.size)
        for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
            oof[val_idx] = self._predict(self.fold_models[fold_id], x_val, *args, **kwargs)
        return oof

    def predict(self, X, *args, **kwargs):
        oof = np.zeros(shape=X.index.size)
        for fold_model in self.fold_models:
            oof += self._predict(fold_model, X, *args, **kwargs) / self.cv_obj.n_splits
        return oof


class BaseCallback:
    def on_before_fit(self, fold_model, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        return fold_model, x_trn, y_trn, x_val, y_val

    def on_after_fit(self, fold_model, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        return fold_model, x_trn, y_trn, x_val, y_val

    def on_before_predict(self, fold_model, X, *args, **kwargs):
        return fold_model, X

    def on_after_predict(self, fold_model, X, results, *args, **kwargs):
        return fold_model, X, results


class ModelClassifierCV(BaseCVWrapper):
    pass
