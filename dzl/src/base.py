import sklearn
from sklearn.model_selection import StratifiedKFold
from tqdm.autonotebook import tqdm
from abc import abstractmethod

import joblib
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm
from typing import Optional, Any
import pandas as pd
import pathlib
import warnings
import numpy as np
import sklearn.preprocessing as prep
import category_encoders as ce
from sklearn import metrics

warnings.filterwarnings('ignore')

scorers = dict(roc_auc_score=metrics.roc_auc_score,
               log_loss=metrics.log_loss)


def create_data_manager(data: pd.DataFrame,
                        cv_column: str,
                        train_split_column: str,
                        label_columns: list,
                        drop_columns=None,
                        weight_column: Optional[str] = None,
                        categorical_features='auto',
                        cv_object=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        ):
    _data = data.copy()

    if cv_column not in _data.columns:
        _data[cv_column] = np.nan

    control_df = _data.loc[:, [train_split_column]]
    label_df = _data.loc[:, label_columns]

    if drop_columns is None:
        drop_columns = []

    if label_columns is None:
        label_columns = []

    assert train_split_column in control_df.columns
    if weight_column:
        assert weight_column in control_df.columns

    feature_columns = [col for col in _data.columns if col not in drop_columns]
    feature_columns = [col for col in feature_columns if col not in control_df.columns]
    feature_columns = [col for col in feature_columns if col not in label_df.columns]

    if isinstance(categorical_features, str):
        if categorical_features == 'auto':
            categorical_features = _data.select_dtypes([object, 'category']).columns.tolist()
        else:
            categorical_features = [categorical_features]

    categorical_features = list(set(categorical_features))

    missing_categorical_columns = list(set(feature_columns) - set(categorical_features))
    assert len(missing_categorical_columns) > 0, f'categorical_features {missing_categorical_columns} are missed'

    if len(label_columns) == 0:
        label_columns = label_df.columns

    assert set(label_columns).issubset(label_df.columns)

    label_columns = [col for col in label_columns if col not in drop_columns]

    columns = feature_columns + label_columns + [cv_column, train_split_column]

    if weight_column:
        columns.append(weight_column)

    return DataManager(data=_data.loc[:, columns],
                       feature_columns=feature_columns,
                       label_columns=label_columns,
                       weight_column=weight_column,
                       cv_column=cv_column,
                       train_split_column=train_split_column,
                       categorical_features=categorical_features
                       ).set_new_cv(cv_object)


encoders_dict = {enc_cls.__name__: enc_cls for enc_cls in [ce.BackwardDifferenceEncoder,
                                                           ce.BaseNEncoder,
                                                           ce.BinaryEncoder,
                                                           ce.CatBoostEncoder,
                                                           ce.CountEncoder,
                                                           ce.GLMMEncoder,
                                                           ce.HashingEncoder,
                                                           ce.HelmertEncoder,
                                                           ce.JamesSteinEncoder,
                                                           ce.LeaveOneOutEncoder,
                                                           ce.MEstimateEncoder,
                                                           ce.OneHotEncoder,
                                                           ce.OrdinalEncoder,
                                                           ce.SumEncoder,
                                                           ce.PolynomialEncoder,
                                                           ce.TargetEncoder,
                                                           ce.WOEEncoder]
                 }


class DataManager:

    def __init__(self,
                 data: pd.DataFrame,
                 feature_columns: list,
                 label_columns: list,
                 cv_column: str,
                 train_split_column: str,
                 categorical_features: list,
                 weight_column: Optional[str] = None,
                 ):
        self.data = data
        self.index_columns = list(data.index.names)
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.weight_column = weight_column
        self.cv_column = cv_column
        self.train_split_column = train_split_column
        self.categorical_features = categorical_features
        self.__active_fold = -1

    @classmethod
    def from_dataframe(cls,
                       data,
                       cv_column,
                       train_split_column,
                       label_columns,
                       drop_columns=None,
                       weight_column: Optional[str] = None,
                       categorical_features='auto',
                       cv_object=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)):
        return create_data_manager(data=data,
                                   cv_column=cv_column,
                                   train_split_column=train_split_column,
                                   label_columns=label_columns,
                                   drop_columns=drop_columns,
                                   weight_column=weight_column,
                                   categorical_features=categorical_features,
                                   cv_object=cv_object)

    @property
    def labeled_mask(self):
        return self.data[self.train_split_column].values.astype(bool).astype(bool)

    @property
    def test_mask(self):
        return ~self.labeled_mask

    @property
    def trn_mask(self):
        return (self.labeled_mask & (self.data[self.cv_column] != self.active_fold).values).astype(np.bool)

    @property
    def val_mask(self):
        return (self.labeled_mask & (self.data[self.cv_column] == self.active_fold).values).astype(np.bool)

    @property
    def active_fold(self):
        return self.__active_fold

    @active_fold.setter
    def active_fold(self, fold_id):
        self.__active_fold = fold_id

    @property
    def X(self):
        return self.data.loc[:, self.feature_columns]

    @property
    def y(self):
        return self.data.loc[:, self.label_columns]

    @property
    def w(self):
        return self.data.loc[:, self.weight_column]

    @property
    def train(self):
        return DataManager(data=self.data.loc[self.trn_mask, :],
                           feature_columns=self.feature_columns,
                           label_columns=self.label_columns,
                           weight_column=self.weight_column,
                           cv_column=self.cv_column,
                           train_split_column=self.train_split_column,
                           categorical_features=self.categorical_features
                           )

    def copy(self):
        return DataManager(data=self.data.copy(),
                           feature_columns=self.feature_columns,
                           label_columns=self.label_columns,
                           weight_column=self.weight_column,
                           cv_column=self.cv_column,
                           train_split_column=self.train_split_column,
                           categorical_features=self.categorical_features
                           )

    @property
    def valid(self):
        return DataManager(data=self.data.loc[self.val_mask, :],
                           feature_columns=self.feature_columns,
                           label_columns=self.label_columns,

                           weight_column=self.weight_column,
                           cv_column=self.cv_column,
                           train_split_column=self.train_split_column,
                           categorical_features=self.categorical_features
                           )

    @property
    def test(self):
        return DataManager(data=self.data.loc[self.test_mask, :],
                           feature_columns=self.feature_columns,
                           label_columns=self.label_columns,
                           weight_column=self.weight_column,
                           cv_column=self.cv_column,
                           train_split_column=self.train_split_column,
                           categorical_features=self.categorical_features
                           )

    @property
    def labeled(self):
        return DataManager(data=self.data.loc[self.labeled_mask, :],
                           feature_columns=self.feature_columns,
                           label_columns=self.label_columns,
                           weight_column=self.weight_column,
                           cv_column=self.cv_column,
                           train_split_column=self.train_split_column,
                           categorical_features=self.categorical_features
                           )

    def __getitem__(self, item):
        if item.lower() in ['train', 'trn']:
            return self.train
        if item.lower() in ['valid', 'val']:
            return self.valid
        if item.lower() in ['test', 'tst']:
            return self.test

    def __hash__(self):
        return int(pd.util.hash_pandas_object(self.data).sum() % 1e8)

    def set_new_cv(self, cv_splitter, groups_column=None):
        self.data[self.cv_column] = np.nan
        if groups_column is not None:
            grp = prep.LabelEncoder().fit_transform(self.labeled.data[groups_column].astype(str).values)
        else:
            grp = None
        s = self.labeled.data[self.cv_column]
        for fold_id, (trn_idx, val_idx) in enumerate(cv_splitter.split(s, self.labeled.y, groups=grp)):
            s.iloc[val_idx] = fold_id
        self.data[self.cv_column] = s
        return self

    def categorical_encode(self, method):

        vals = []
        tsts = []
        for fold_id in self.data[self.cv_column].dropna().sort_values().unique():
            self.active_fold = fold_id
            encoder = encoders_dict[method](self.categorical_features)
            encoder.fit(self.train.data, self.train.y)
            vals.append(encoder.transform(self.valid.data))
            tsts.append(encoder.transform(self.test.data))

        data = pd.concat([pd.concat(vals), pd.concat(tsts).groupby(level=0).mean()]).sort_index()

        return DataManager(data=data,
                           feature_columns=self.feature_columns,
                           label_columns=self.label_columns,
                           weight_column=self.weight_column,
                           cv_column=self.cv_column,
                           train_split_column=self.train_split_column,
                           categorical_features=self.categorical_features
                           )


class CVTrainer:
    def __init__(self,
                 fold_trainer_cls,
                 model_name: str,
                 params: dict, ds: DataManager,
                 save_path: pathlib.Path,
                 callbacks: list = [],
                 task='predict_proba',
                 *args, **kwargs):
        self.fold_trainer_cls = fold_trainer_cls
        self.params = params
        self.n_folds: int = int(ds.data[ds.cv_column].max() + 1)
        self.task = task
        self.args = args
        self.kwargs = kwargs
        self.model_name = model_name
        self.save_path = save_path / 'models' / self.model_name
        self.ds = ds
        self.callbacks = callbacks
        self.trainers: list = [self.fold_trainer_cls(model_name=model_name,
                                                     ds=ds,
                                                     fold_id=fold_id,
                                                     save_path=self.save_path,
                                                     params=self.params,
                                                     task=self.task,
                                                     *self.args,
                                                     **self.kwargs)
                               for fold_id in range(self.n_folds)]

    def fit(self):
        for trainer in tqdm(self.trainers):
            trainer.fit()
        return self

    def fit_single_fold(self):
        trainer = self.trainers[0]
        trainer.fit()
        return trainer.score('val')

    def save(self, run_name='', is_final_model=True):
        params_path = self.save_path / f'{self.model_name}_params.pkl'

        if is_final_model:
            score = self.score('val')
            if run_name:
                output_name = self.save_path.parent / f'{score:.5f}_{self.model_name}_{run_name}.csv'
            else:
                output_name = self.save_path.parent / f'{score:.5f}_{self.model_name}.csv'
            if not output_name.exists():
                (self
                 .predict('tst')
                 .groupby(level=0)
                 .mean()
                 .reset_index()
                 .to_csv(output_name, index=False)
                 )

        if run_name:
            out_features_path = self.save_path / f'{self.model_name}_{run_name}.csv'
        else:
            out_features_path = self.save_path / f'{self.model_name}.csv'
        if not out_features_path.exists():
            (self
             .extract_features()
             .data.reset_index()
             .to_csv(out_features_path, index=False)
             )

        if not params_path.exists():
            joblib.dump(self.params, params_path)
        return self

    def load(self):
        for trainer in self.trainers:
            trainer.load()
        return self

    def extract_features(self) -> pd.DataFrame:
        index = self.ds.index_columns
        data = pd.concat([self.predict('val'),
                          self.predict('tst').groupby(level=index).mean()])
        data.columns = [f'{self.model_name}__{column}' for column in data.columns]
        return data

    def predict(self, typ: str):
        results = []
        for trainer in self.trainers:
            results.append(trainer.predict(typ=typ))
        return pd.concat(results)

    def score(self, typ: str):
        metric = self.trainers[0].metric

        ypred, ytrue = self.predict(typ).align(self.ds.labeled.y, join='right')
        return scorers[metric](ytrue, ypred)


class CVVoteTrainer(CVTrainer):
    def extract_features(self) -> pd.DataFrame:
        index = self.ds.index_columns
        data = pd.concat([self.predict('val'),
                          self.predict('tst').groupby(level=index).agg(lambda x: x.value_counts().index[0])])
        data.columns = [f'{self.model_name}__{column}' for column in data.columns]
        return data


class Callback:
    def __init__(self):
        pass

    def before_fit(self, trainer):
        pass

    def after_fit(self, trainer):
        pass


class BaseFoldTrainer:

    def __init__(self,
                 model_name: str,
                 ds: DataManager,
                 fold_id: int,
                 save_path: pathlib.Path,
                 params: dict,
                 ext: str = 'model',
                 metric='roc_auc_score',
                 task='predict_proba'):
        self.fold_id: int = fold_id
        self.model_name: str = model_name
        self.model_fname = f"{ds.cv_column}_fold{fold_id}.{ext}"
        self.ds: DataManager = ds
        self.model = None
        self.save_path: pathlib.Path = save_path / self.model_fname
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: list = []
        self.params: dict = params
        self.fit = self.fit_decor(self.fit)
        self.predict = self.predict_decor(self.predict)
        self.extract_features = self.extract_decor(self.extract_features)
        self.metric = metric
        self.task = task

    def fit_decor(self, fn):
        def inner(*args, **kwargs):
            self.ds.active_fold = self.fold_id
            if self.exists():
                print(f'model {self.model_name} already trained, loading pretrained')
                self.load()
                return

            fn(*args, **kwargs)
            self.save()
            return

        return inner

    def predict_decor(self, fn):
        def inner(*args, **kwargs):
            self.ds.active_fold = self.fold_id
            if not self.exists():
                raise Exception(f'model {self.model_name} not trained yet')
            ret = fn(*args, **kwargs)
            return ret

        return inner

    def extract_decor(self, fn):
        def inner(*args, **kwargs):
            self.ds.active_fold = self.fold_id
            if not self.exists():
                raise Exception(f'model {self.model_name} not trained yet')
            ret = fn(*args, **kwargs)
            return ret

        return inner

    @abstractmethod
    def get_model(self):
        return

    @abstractmethod
    def fit(self):
        return self

    @abstractmethod
    def predict(self, typ: str) -> pd.DataFrame:
        return pd.DataFrame()

    @abstractmethod
    def extract_features(self, typ: str) -> pd.DataFrame:
        return pd.DataFrame()

    def score(self, typ: str):
        ypred, ytrue = self.predict(typ).align(self.ds[typ].y, join='right')

        score = scorers[self.metric](ytrue, ypred)
        print(f'{self.metric} {typ}: {score}')
        return score

    def exists(self):
        return self.save_path.exists()

    def save(self):
        joblib.dump(self.model, str(self.save_path))
        return self

    def load(self):
        self.model = joblib.load(str(self.save_path))
        return self


class BaseFoldClassifier(BaseFoldTrainer):
    def predict(self, typ: str):
        preds = pd.DataFrame(self.model.predict_proba(self.ds[typ].X.values)[:, 1],
                             index=self.ds[typ].X.index,
                             columns=self.ds[typ].y.columns)
        return preds


class BaseFoldRegressor(BaseFoldTrainer):
    def predict(self, typ: str):
        preds = pd.DataFrame(self.model.predict(self.ds[typ].X.values),
                             index=self.ds[typ].X.index,
                             columns=self.ds[typ].y.columns)
        return preds
