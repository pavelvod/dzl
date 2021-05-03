import numpy as np


class BaseCVClassifierWrapper:

    def __init__(self, model_cls, model_params, cv_obj, *args, **kwargs):
        self.cv_obj = cv_obj
        self.model_params = model_params
        self.models = [model_cls(**self.model_params) for fold_id in range(self.cv_obj.n_splits)]
        self._classes = None
        self.__fitted = False

    def _fit(self, model, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        model.fit(x_trn, y_trn, *args, **kwargs)
        return self

    def _predict_proba(self, model, X, *args, **kwargs):
        return model.predict_proba(X, *args, **kwargs)

    def generate_folds(self, X, y):
        for fold_id, (trn_idx, val_idx) in enumerate(self.cv_obj.split(X, y)):
            x_trn, x_val = X.iloc[trn_idx, :], X.iloc[val_idx, :]
            y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
            yield fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val

    def fit(self, X, y, *args, **kwargs):
        self._classes = np.unique(y)
        for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
            self._fit(self.models[fold_id], x_trn, y_trn, x_val, y_val, *args, **kwargs)
        return self

    def predict_proba_trn(self, X, y, *args, **kwargs):
        oof = np.zeros(shape=(X.index.size, len(self._classes)))
        for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
            oof[trn_idx, :] += self._predict_proba(self.models[fold_id], x_trn, *args, **kwargs) / (
                    self.cv_obj.n_splits - 1)
        return oof

    def predict_proba_val(self, X, y, *args, **kwargs):
        oof = np.zeros(shape=(X.index.size, len(self._classes)))
        for fold_id, trn_idx, val_idx, x_trn, y_trn, x_val, y_val in self.generate_folds(X, y):
            oof[val_idx, :] = self._predict_proba(self.models[fold_id], x_val, *args, **kwargs)
        return oof

    def predict_proba(self, X, *args, **kwargs):
        oof = np.zeros(shape=(X.index.size, len(self._classes)))
        for model in self.models:
            oof += self._predict_proba(model, X, *args, **kwargs) / self.cv_obj.n_splits
        return oof


class LGBMCVClassifierWrapper(BaseCVClassifierWrapper):

    def _fit(self, model, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        model.fit(x_trn, y_trn,
                  eval_set=[(x_trn, y_trn), (x_val, y_val)],
                  *args, **kwargs)
        return self
