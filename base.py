import pathlib
from abc import abstractmethod
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class DataManager:
    def __init__(self,
                 data,
                 feature_columns,
                 label_columns,
                 index_columns,
                 drop_columns,
                 weight_column,
                 filter_column,
                 cv_column,
                 train_split_column):
        self.data = data
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.index_columns = index_columns
        self.drop_columns = drop_columns
        self.weight_column = weight_column
        self.filter_column = filter_column
        self.cv_column = cv_column
        self.train_split_column = train_split_column

        self.__active_fold = -1

        self.prepare()

    def prepare(self):
        not_feature_columns = self.index_columns + self.drop_columns + [self.weight_column, self.cv_column,
                                                                        self.filter_column, self.train_split_column]
        self.label_columns = list(set(self.label_columns) - set(not_feature_columns))
        self.feature_columns = list(set(self.feature_columns) - set(not_feature_columns) - set(self.label_columns))

    @property
    def labeled_mask(self):
        return self.data[self.train_split_column].values

    @property
    def test_mask(self):
        return ~self.labeled_mask

    @property
    def filter_mask(self):
        return self.data[self.filter_column].values

    @property
    def trn_mask(self):
        return self.labeled_mask & self.filter_mask & (self.data[self.cv_column] != self.active_fold).values

    @property
    def val_mask(self):
        return self.filter_mask & (self.data[self.cv_column] == self.active_fold).values

    @property
    def active_fold(self):
        return self.__active_fold

    @active_fold.setter
    def active_fold(self, fold_id):
        self.__active_fold = fold_id

    @property
    def X(self):
        return self.data.set_index(self.index_columns).loc[:, self.feature_columns]

    @property
    def y(self):
        return self.data.set_index(self.index_columns).loc[:, self.label_columns]

    @property
    def w(self):
        return self.data.set_index(self.index_columns).loc[:, self.weight_column]

    # @property
    # def catboost(self):
    #     return catboost.Pool(data=self.X,
    #                          label=self.y,
    #                          weight=self.w)

    @property
    def train(self):
        return DataManager(self.data.loc[self.trn_mask, :],
                           self.feature_columns,
                           self.label_columns,
                           self.index_columns,
                           self.drop_columns,
                           self.weight_column,
                           self.filter_column,
                           self.cv_column,
                           self.train_split_column)

    @property
    def valid(self):
        return DataManager(self.data.loc[self.val_mask, :],
                           self.feature_columns,
                           self.label_columns,
                           self.index_columns,
                           self.drop_columns,
                           self.weight_column,
                           self.filter_column,
                           self.cv_column,
                           self.train_split_column)

    @property
    def test(self):
        return DataManager(self.data.loc[self.test_mask, :],
                           self.feature_columns,
                           self.label_columns,
                           self.index_columns,
                           self.drop_columns,
                           self.weight_column,
                           self.filter_column,
                           self.cv_column,
                           self.train_split_column)

    @property
    def labeled(self):
        return DataManager(self.data.loc[self.labeled_mask, :],
                           self.feature_columns,
                           self.label_columns,
                           self.index_columns,
                           self.drop_columns,
                           self.weight_column,
                           self.filter_column,
                           self.cv_column,
                           self.train_split_column)

    def __getitem__(self, item):
        if item.lower() in ['train', 'trn']:
            return self.train
        if item.lower() in ['valid', 'val']:
            return self.valid
        if item.lower() in ['test', 'tst']:
            return self.test


class BaseFoldTrainer:
    def __init__(self, model_name: str, seed: int, fold_id: int, save_path: pathlib.Path, params: dict):
        self.fold_id: int = fold_id
        self.model_name: str = model_name
        self.seed: int = seed
        self.model = None
        self.save_path: pathlib.Path = save_path
        self.history: list = []
        self.params: dict = params

    @abstractmethod
    def fit(self, ds: DataManager):
        ds.active_fold = self.fold_id
        return self

    @abstractmethod
    def predict(self, ds: DataManager, typ: str) -> pd.DataFrame:
        ds.active_fold = self.fold_id
        return pd.DataFrame()

    @abstractmethod
    def extract_features(self, ds: DataManager, typ: str) -> pd.DataFrame:
        ds.active_fold = self.fold_id
        return pd.DataFrame()

    @abstractmethod
    def score(self, ds: DataManager, typ: str) -> pd.DataFrame:
        ds.active_fold = self.fold_id
        return pd.Dataframe()

    @abstractmethod
    def save(self):
        return self

    @abstractmethod
    def load(self):
        return self


class CVTrainer:
    def __init__(self, fold_trainer_cls, n_folds: int, model_name: str, seed: int, params: dict,
                 save_path: pathlib.Path, *args, **kwargs):
        self.fold_trainer_cls = fold_trainer_cls
        self.params = params
        self.n_folds: int = n_folds
        self.save_path = save_path
        self.args = args
        self.kwargs = kwargs
        self.trainers: list = [self.fold_trainer_cls(model_name=model_name,
                                                     seed=seed,
                                                     fold_id=fold_id,
                                                     save_path=self.save_path,
                                                     params=self.params, *self.args,
                                                     **self.kwargs)
                               for fold_id in range(self.n_folds)]

    def fit(self, ds: DataManager, save: bool = True):
        for trainer in self.trainers:
            trainer.fit(ds)
            if save:
                trainer.save()
        return self

    def save(self):
        for trainer in self.trainers:
            trainer.save()
        return self

    def load(self):
        for trainer in self.trainers:
            trainer.load()
        return self

    def extract_features(self, ds: DataManager, typ: str) -> pd.DataFrame:
        results = []
        for trainer in self.trainers:
            results.append(trainer
                           .extract_features(ds=ds, typ=typ)
                           .assign(fold_id=trainer.fold_id)
                           )
        return pd.concat(results)

    def predict(self, ds: DataManager, typ: str):
        results = []
        for trainer in self.trainers:
            results.append(trainer.predict(ds=ds, typ=typ))
        return pd.concat(results)

    def score(self, ds: DataManager, typ: str):
        return {trainer.fold_id: trainer.score(ds=ds, typ=typ) for trainer in self.trainers}
