import lightgbm as lightgbm

import pandas as pd

from .base import BaseFoldRegressor


class LightGBMRegressorFoldTrainer(BaseFoldRegressor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self):
        return lightgbm.LGBMRegressor(**self.params['init_params'])

    def fit(self):
        print(self.ds.active_fold, self.ds.train.X.shape, self.ds.train.y.shape, self.ds.valid.X.shape,
              self.ds.valid.y.shape)
        self.model = self.get_model()
        print(self.ds.train.X.select_dtypes('object').columns.tolist())
        self.model.fit(self.ds.train.X,
                       y=self.ds.train.y,
                       categorical_feature=self.ds.train.X.select_dtypes('object').columns.tolist(),
                       eval_set=[(self.ds.valid.X, self.ds.valid.y)],
                       **self.params['fit_params']
                       )
        return self

    def predict(self, typ: str):
        preds = pd.DataFrame(self.model.predict(self.ds[typ].X), index=self.ds[typ].X.index,
                             columns=self.ds[typ].y.columns)
        return preds

    def extract_features(self, typ: str) -> pd.DataFrame:
        return self.predict(typ=typ)



