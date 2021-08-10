from dzl.utils import Task
import numpy as np
import pandas as pd


class BaseCallback:
    def __init__(self):
        pass

    def on_before_fit(self, model, X, y, sample_weight, *args, **kwargs):
        return X, y, sample_weight

    def on_after_fit(self, model, X, y, *args, **kwargs):
        return

    def on_before_fold_fit(self, model, fold_model, trn_idx, val_idx, x_trn, y_trn, w_trn, x_val, y_val, w_val, *args,
                           **kwargs):
        return fold_model, x_trn, y_trn, x_val, y_val

    def on_after_fold_fit(self, model, fold_model, trn_idx, val_idx, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        return

    def on_before_fold_predict(self, model, fold_model, X, x_trn, *args, **kwargs):
        return x_trn

    def on_after_fold_predict(self, model, fold_model, X, x_trn, results, *args, **kwargs):
        return results


class OOFValidCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.oof = None
        self.n_seeds = None
        self.task = None

    def on_before_fit(self, model, X, y, sample_weight, *args, **kwargs):
        self.task = Task.Classification if 'predict_proba' in dir(model.model_cls) else Task.Regression
        self.n_seeds = len(model.seeds)

        if self.task == Task.Regression:
            self.oof = np.zeros(shape=X.shape[0])
        if self.task == Task.Classification:
            self.oof = np.zeros(shape=(X.shape[0], y.nunique()))
        return X, y, sample_weight

    def on_after_fold_fit(self, model, fold_model, trn_idx, val_idx, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        if self.task == Task.Regression:
            self.oof[val_idx] += fold_model.predict(x_val) / self.n_seeds
        if self.task == Task.Classification:
            self.oof[val_idx, :] += fold_model.predict_proba(x_val) / self.n_seeds
        return


class FoldMetricCallback(BaseCallback):
    def __init__(self, metric_list: list):
        super().__init__()
        self.metric_list = metric_list
        self.n_seeds = None
        self.task = None

    def on_before_fit(self, model, X, y, sample_weight, *args, **kwargs):
        self.task = Task.Classification if 'predict_proba' in dir(model.model_cls) else Task.Regression
        self.n_seeds = len(model.seeds)
        return X, y, sample_weight

    def on_after_fold_fit(self, model, fold_model, trn_idx, val_idx, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        print({metric.__name__: metric(y_val, fold_model.predict_proba(x_val)[:, 1]) for metric in self.metric_list})


class FoldMultiClassMetricCallback(BaseCallback):
    def __init__(self, metric_list: list):
        super().__init__()
        self.metric_list = metric_list
        self.n_seeds = None
        self.task = None

    def on_before_fit(self, model, X, y, sample_weight, *args, **kwargs):
        self.task = Task.Classification if 'predict_proba' in dir(model.model_cls) else Task.Regression
        self.n_seeds = len(model.seeds)
        return X, y, sample_weight

    def on_after_fold_fit(self, model, fold_model, trn_idx, val_idx, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        print({metric.__name__: metric(y_val, fold_model.predict_proba(x_val)) for metric in self.metric_list})


class TabNetCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def on_before_fold_fit(self, model, fold_model, trn_idx, val_idx, x_trn, y_trn, w_trn, x_val, y_val, w_val, *args,
                           **kwargs):
        if w_trn is not None:
            return fold_model, x_trn.values, y_trn.values, w_trn.values, x_val.values, y_val.values, w_val.values
        else:
            return fold_model, x_trn.values, y_trn.values, x_val.values, y_val.values


class CatBoostFeatureImportanceCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.importances: list = []

    def on_before_fit(self, model, X, y, sample_weight, *args, **kwargs):
        return X, y, sample_weight

    def on_after_fold_fit(self, model, fold_model, trn_idx, val_idx, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        self.importances.append(dict(zip(fold_model.feature_names_, fold_model.feature_importances_)))

    def finalize(self):
        df = pd.DataFrame(self.importances).T
        return df.mean(1).subtract(df.std(1).mul(2)).sort_values(ascending=False)
