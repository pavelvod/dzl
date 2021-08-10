import copy
import inspect
from collections import defaultdict
from functools import reduce
from typing import Optional

import catboost
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold


class BaseCVWrapper:
    def __init__(self,
                 model_cls,
                 model_params,
                 fit_params: Optional[dict] = None,
                 cv_cls=None,
                 n_folds: int = 5,
                 cv_params: Optional[dict] = None,
                 seeds: Optional[list] = None,
                 callbacks: Optional[list] = None,
                 groups=None,

                 *args, **kwargs):
        self.n_folds: int = n_folds
        self.fit_params: dict = fit_params or {}
        self.cv_params: dict = cv_params
        self.seeds: list = seeds or [42]
        self.model_params: dict = model_params
        self.model_cls = model_cls

        if cv_cls is None:
            if 'predict_proba' in dir(model_cls):
                cv_cls = StratifiedKFold
            else:
                cv_cls = KFold

        self.cv_cls = cv_cls
        self.groups = groups
        self.fold_models: defaultdict = defaultdict(dict)
        self.callbacks = callbacks or []

    # utils
    def get_cv_obj(self, seed):
        lst_of_param_names = inspect.getfullargspec(self.cv_cls.__init__).args
        if self.cv_params:
            cv_params = copy.deepcopy(self.cv_params)
        else:
            cv_params = dict()
        cv_params['n_splits'] = self.n_folds
        if 'random_state' in lst_of_param_names:
            cv_params['random_state'] = seed

        if 'shuffle' in lst_of_param_names:
            cv_params['shuffle'] = True

        return self.cv_cls(**cv_params)

    def generate_folds(self, X, y, sample_weight):
        for seed in self.seeds:
            cv_obj = self.get_cv_obj(seed)
            for fold_id, (trn_idx, val_idx) in enumerate(cv_obj.split(X, y, groups=self.groups)):
                x_trn, x_val = X.iloc[trn_idx, :], X.iloc[val_idx, :]
                y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
                w_trn, w_val = sample_weight.iloc[trn_idx], sample_weight.iloc[val_idx]
                yield seed, fold_id, trn_idx, val_idx, x_trn, y_trn, w_trn, x_val, y_val, w_val

    def fold_models_flatten(self):
        return reduce(lambda x, y: x + y, [list(d.values()) for d in self.fold_models.values()])

    def __fit(self, fold_model, x_trn, y_trn, w_trn, x_val, y_val, w_val, *args, **kwargs):
        for k, v in kwargs.items():
            self.fit_params[k] = v

        if fold_model.__module__ == 'catboost.core':
            trn_pool = catboost.Pool(data=x_trn, label=y_trn, weight=w_trn)
            val_pool = catboost.Pool(data=x_val, label=y_val, weight=w_val)
            fold_model.fit(X=trn_pool, eval_set=[trn_pool, val_pool], **self.fit_params)
            return fold_model

        if fold_model.__module__ == 'lightgbm.sklearn':
            fold_model.fit(x_trn, y_trn,
                           eval_set=[(x_val, y_val)],
                           sample_weight=w_trn,
                           eval_sample_weight=w_val,
                           **self.fit_params)

            return fold_model

        if 'eval_set' in inspect.getfullargspec(fold_model.fit).args:
            fold_model.fit(x_trn, y_trn,
                           eval_set=[(x_val, y_val)],
                           **self.fit_params)
            return fold_model

        fold_model.fit(x_trn, y_trn, sample_weight=w_trn, *args, **kwargs)
        return fold_model

    # any task
    def _fit(self, fold_model, trn_idx, val_idx, x_trn, y_trn, w_trn, x_val, y_val, w_val, *args, **kwargs):
        for callback in self.callbacks:
            fold_model, x_trn, y_trn, w_trn, x_val, y_val, w_val = callback.on_before_fold_fit(self, fold_model,
                                                                                               trn_idx, val_idx,
                                                                                               x_trn, y_trn, w_trn,
                                                                                               x_val, y_val, w_val,
                                                                                               *args,
                                                                                               **kwargs)

            fold_model = self.__fit(fold_model, x_trn, y_trn, w_trn, x_val, y_val, w_val, *args, **kwargs)

            for callback in self.callbacks:
                callback.on_after_fold_fit(self, fold_model, trn_idx, val_idx, x_trn, y_trn, x_val, y_val, *args,
                                           **kwargs)
            return self

    def fit(self, X, y, sample_weight=None, *args, **kwargs):
        for callback in self.callbacks:
            X, y, sample_weight = callback.on_before_fit(self, X, y, sample_weight, *args, **kwargs)

        for tmp in self.generate_folds(X, y, sample_weight):
            seed, fold_id, trn_idx, val_idx, x_trn, y_trn, w_trn, x_val, y_val, w_val = tmp
            fold_model = self.model_cls(**self.model_params)
            self.fold_models[seed][fold_id] = fold_model
            self._fit(fold_model, trn_idx, val_idx, x_trn, y_trn, w_trn, x_val, y_val, w_val, *args, **kwargs)

        for callback in self.callbacks:
            callback.on_after_fit(self, X, y, *args, **kwargs)
        return self


class ModelClassifierCV(BaseCVWrapper):

    def __init__(self,
                 model_cls,
                 model_params,
                 fit_params: Optional[dict] = None,
                 cv_cls=None,
                 n_folds: int = 5,
                 cv_params: Optional[dict] = None,
                 seeds: Optional[list] = None,
                 callbacks: Optional[list] = None,
                 groups=None,
                 return_logits: bool = False, *args, **kwargs):
        super().__init__(model_cls=model_cls,
                         model_params=model_params,
                         fit_params=fit_params,
                         cv_cls=cv_cls,
                         n_folds=n_folds,
                         cv_params=cv_params,
                         seeds=seeds,
                         callbacks=callbacks,
                         groups=groups,
                         *args, **kwargs)
        self._classes = None
        self.return_logits = return_logits
        self.eps = 1e-8

    def fit(self, X, y, sample_weight, *args, **kwargs):
        self._classes = np.unique(y)
        super().fit(X, y, sample_weight, *args, **kwargs)
        return self

    def _predict_proba(self, fold_model, X, *args, **kwargs):
        out = fold_model.predict_proba(X, *args, **kwargs)
        if self.return_logits:
            out = out.clip(self.eps, 1 - self.eps)
            out = np.log(out / (1 - out))
        return out

    def predict_proba(self, X, *args, **kwargs):
        oof = np.zeros(shape=(X.index.size, len(self._classes)))
        lst_models = self.fold_models_flatten()
        for fold_model in lst_models:
            oof += self._predict_proba(fold_model, X, *args, **kwargs) / (len(lst_models))
        return oof

    def predict(self, X, *args, **kwargs):
        return self.predict_proba(X, *args, **kwargs).argmax(1)


class ModelRegressorCV(BaseCVWrapper):

    def _predict(self, fold_model, X, *args, **kwargs):
        x_trn = X.copy()
        for callback in self.callbacks:
            x_trn = callback.on_before_fold_predict(self, fold_model, X, x_trn, *args, **kwargs)

        results = fold_model.predict(x_trn, *args, **kwargs)

        for callback in self.callbacks:
            results = callback.on_after_fold_predict(self, fold_model, X, x_trn, results, *args, **kwargs)

        return results

    def predict(self, X, *args, **kwargs):
        oof = np.zeros(shape=X.index.size)
        lst_models = self.fold_models_flatten()
        for fold_model in lst_models:
            oof += self._predict(fold_model, X, *args, **kwargs) / (len(lst_models))
        return oof


from catboost import cv
