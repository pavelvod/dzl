import catboost
import lightgbm as lightgbm
import xgboost
import pandas as pd

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier

from base import BaseFoldClassifier


class CatboostClassifierFoldTrainer(BaseFoldClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        return catboost.CatBoostClassifier(**self.params['init_params'])

    def fit(self):
        # print(self.ds.active_fold, self.ds.train.X.shape, self.ds.train.y.shape, self.ds.valid.X.shape,
        #       self.ds.valid.y.shape)
        self.model = self.get_model()
        print(self.params)
        # print(self.ds.train.X.select_dtypes('object').columns.tolist())
        self.model.fit(self.ds.train.X,
                       y=self.ds.train.y,
                       cat_features=self.ds.categorical_features,
                       eval_set=[(self.ds.valid.X, self.ds.valid.y)],
                       **self.params['fit_params']
                       )

        return self

    def predict(self, typ: str):
        preds = pd.DataFrame(self.model.predict_proba(self.ds[typ].X.values)[:, 1],
                             index=self.ds[typ].X.index,
                             columns=self.ds[typ].y.columns)
        return preds

    def extract_features(self, typ: str) -> pd.DataFrame:
        return self.predict(typ=typ)

    def save(self):
        self.model.save_model(str(self.save_path))
        return self

    def load(self):
        self.model = self.get_model()
        self.model.load_model(str(self.save_path))
        return self


class LightGBMClassifierFoldTrainer(BaseFoldClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        return lightgbm.LGBMClassifier(**self.params['init_params'])

    def fit(self):
        print(self.ds.active_fold, self.ds.train.X.shape, self.ds.train.y.shape, self.ds.valid.X.shape,
              self.ds.valid.y.shape)

        self.model = self.get_model()
        self.model.fit(self.ds.train.X,
                       y=self.ds.train.y,
                       categorical_feature='auto',
                       eval_set=[(self.ds.valid.X, self.ds.valid.y)],
                       **self.params['fit_params']
                       )
        return self

    def predict(self, typ: str):
        preds = pd.DataFrame(self.model.predict_proba(self.ds[typ].X.values)[:, 1],
                             index=self.ds[typ].X.index,
                             columns=self.ds[typ].y.columns)
        return preds

    def extract_features(self, typ: str) -> pd.DataFrame:
        return self.predict(typ=typ)


class XGBoostClassifierFoldTrainer(BaseFoldClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        return xgboost.XGBClassifier(**self.params['init_params'])

    def fit(self):
        print(self.ds.active_fold, self.ds.train.X.shape, self.ds.train.y.shape, self.ds.valid.X.shape,
              self.ds.valid.y.shape)
        self.model = self.get_model()

        self.model.fit(self.ds.train.X,
                       y=self.ds.train.y,
                       eval_set=[(self.ds.valid.X, self.ds.valid.y)],
                       early_stopping_rounds=50,
                       **self.params['fit_params']
                       )
        return self

    def predict(self, typ: str):
        preds = pd.DataFrame(self.model.predict_proba(self.ds[typ].X.values)[:, 1],
                             index=self.ds[typ].X.index,
                             columns=self.ds[typ].y.columns)
        return preds

    def extract_features(self, typ: str) -> pd.DataFrame:
        return self.predict(typ=typ)


class TabNetFoldTrainer(BaseFoldClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        return TabNetClassifier(**self.params['init_params'])

    def fit(self):
        print(self.ds.active_fold, self.ds.train.X.shape, self.ds.train.y.shape, self.ds.valid.X.shape,
              self.ds.valid.y.shape)

        self.model = self.get_model()
        print(self.model)
        fit_params = dict(X_train=self.ds.train.X.values,
                          y_train=self.ds.train.y.values.ravel(),
                          eval_set=[(self.ds.train.X.values, self.ds.train.y.values.ravel()),
                                    (self.ds.valid.X.values, self.ds.valid.y.values.ravel())],
                          eval_name=['trn', 'val'],
                          eval_metric=['auc'], )

        if self.params.get('config', {}).get('pretrain', False):
            unsupervised_model = TabNetPretrainer(**self.params['init_params'])
            unsupervised_model.fit(X_train=self.ds.train.X.values,
                                   eval_set=[self.ds.valid.X.values],
                                   pretraining_ratio=0.8
                                   )

            self.model.fit(from_unsupervised=unsupervised_model, **fit_params
                           )
        else:
            self.model.fit(**fit_params)
        return self

    def predict(self, typ: str):
        preds = pd.DataFrame(self.model.predict_proba(self.ds[typ].X.values)[:, 1],
                             index=self.ds[typ].X.index,
                             columns=self.ds[typ].y.columns)
        return preds

    def extract_features(self, typ: str) -> pd.DataFrame:
        return self.predict(typ=typ)


class SklearnClassifierFoldTrainer(BaseFoldClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        import sklearn

        steps = []
        for estimator, params in self.params['pipeline'].items():
            steps.append(eval(estimator)(**params))

        return sklearn.pipeline.make_pipeline(*steps)

    def fit(self):
        # print(self.ds.active_fold, self.ds.train.X.shape, self.ds.train.y.shape, self.ds.valid.X.shape,
        #       self.ds.valid.y.shape)
        self.model = self.get_model()
        self.model.fit(self.ds.train.X,
                       y=self.ds.train.y
                       )
        return self

    def predict(self, typ: str):
        preds = pd.DataFrame(self.model.predict_proba(self.ds[typ].X.values)[:, 1],
                             index=self.ds[typ].X.index,
                             columns=self.ds[typ].y.columns)
        return preds

    def extract_features(self, typ: str) -> pd.DataFrame:
        return self.predict(typ=typ)
