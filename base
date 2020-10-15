import pathlib
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import random
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore')

from itertools import combinations

import numpy as np
import torch


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """

    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        # labels = labels.cpu().data.numpy()
        lbls = labels.cpu().detach().numpy()
        bs = lbls.shape[0]
        labels = -np.ones(shape=bs)
        for a, b in zip(*np.where(lbls == 1)):
            labels[a] = b
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector
        self.criterion = nn.CosineEmbeddingLoss(margin=margin, reduction='sum')

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()

        bs = positive_pairs.shape[0]
        n_pairs = positive_pairs.shape[0] + negative_pairs.shape[0]
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        # pos_loss = self.criterion(embeddings[positive_pairs[:, 0]],
        #                embeddings[positive_pairs[:, 1]],
        #                torch.from_numpy(np.array([1] * bs)))
        #
        # neg_loss = self.criterion(embeddings[negative_pairs[:, 0]],
        #                embeddings[negative_pairs[:, 1]],
        #                torch.from_numpy(np.array([0] * bs)))
        #
        # return (pos_loss + neg_loss) / n_pairs

        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).mean(1)
        negative_loss = torch.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).mean(1))
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(42)


class MoADataSet(torch.utils.data.Dataset):
    def __init__(self, features, targets=pd.DataFrame([])):
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        out = dict(inputs=torch.from_numpy(self.features.iloc[idx, :].values))
        if self.targets.index.size > 0:
            out['targets'] = torch.from_numpy(self.targets.iloc[idx, :].values)
        return out


class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.25)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

        self.ocl_criterion = OnlineContrastiveLoss(margin=0.2, pair_selector=HardNegativePairSelector(cpu=False))

    def extract_features(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = torch.nn.functional.relu(self.dense2(x))
        return x

    def forward(self, inputs, labels):
        embeddings = self.extract_features(inputs)
        x = self.batch_norm3(embeddings)
        x = self.dropout3(x)
        x = self.dense3(x)
        ocloss = self.ocl_criterion(embeddings, labels)
        return x, ocloss


SEED = 42


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
    def __init__(self, fold_id: int, save_path: pathlib.Path):
        self.fold_id: int = fold_id
        self.model = None
        self.save_path: pathlib.Path = save_path
        self.history: list = []

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
    def save(self):
        return self

    @abstractmethod
    def load(self):
        return self


class FoldTrainer(BaseFoldTrainer):
    def __init__(self, fold_id: int, save_path: pathlib.Path):
        super().__init__(fold_id=fold_id, save_path=save_path / f"{fold_id}.pth")

        self.lr = 1e-1
        self.weight_decay = 1e-6
        self.batch_size = 256
        self.epochs = 20
        self.embedding_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCEWithLogitsLoss()

    @staticmethod
    def train_epoch(model, optimizer, scheduler, criterion, dataloader, device):
        model.train()
        final_loss = 0
        oc_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, targets = batch['inputs'].to(device), batch['targets'].to(device)
            outputs, ocloss = model(inputs, targets)
            _loss = criterion(outputs, targets)
            loss = _loss + ocloss
            loss.backward()
            optimizer.step()
            scheduler.step()
            final_loss += _loss.item()
            oc_loss += ocloss.item()

        final_loss /= len(dataloader)
        oc_loss /= len(dataloader)

        return final_loss, oc_loss

    @staticmethod
    def validate_epoch(model, criterion, dataloader, device):
        model.eval()
        final_loss = 0
        oc_loss = 0
        for batch in dataloader:
            inputs, targets = batch['inputs'].to(device), batch['targets'].to(device)
            outputs, ocloss = model(inputs, targets)
            _loss = criterion(outputs, targets)
            final_loss += _loss.item()
            oc_loss += ocloss.item()

        final_loss /= len(dataloader)
        oc_loss /= len(dataloader)

        return final_loss, oc_loss

    def fit(self, ds: DataManager):
        ds.active_fold = self.fold_id
        best = 9999999
        trn_dl = torch.utils.data.DataLoader(MoADataSet(ds.train.X, ds.train.y), batch_size=self.batch_size)
        val_dl = torch.utils.data.DataLoader(MoADataSet(ds.valid.X, ds.valid.y), batch_size=self.batch_size)

        self.model = Model(ds.train.X.shape[1], ds.train.y.shape[1], self.embedding_size).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                  pct_start=0.05,
                                                  div_factor=1.5e3,
                                                  max_lr=1e-2,
                                                  epochs=self.epochs,
                                                  steps_per_epoch=len(trn_dl)
                                                  )

        for epoch in tqdm(range(self.epochs)):
            trn_loss, trn_ocloss = self.train_epoch(self.model, optimizer, scheduler, self.criterion, trn_dl,
                                                    self.device)
            val_loss, val_ocloss = self.validate_epoch(self.model, self.criterion, val_dl, self.device)
            logs = dict(fold_id=self.fold_id, epoch=epoch, trn_loss=trn_loss, val_loss=val_loss, trn_ocloss=trn_ocloss,
                        val_ocloss=val_ocloss)
            print('\t'.join([f'{k}={v:.4f}' for k, v in logs.items()]))
            self.history.append(logs)
            if best > val_loss:
                best = val_loss
                torch.save(self.model.state_dict(), self.save_path)
        self.model.load_state_dict(torch.load(self.save_path))
        return self

    def predict(self, ds: DataManager, typ: str):
        ds.active_fold = self.fold_id
        dl = torch.utils.data.DataLoader(MoADataSet(ds[typ].X, ds[typ].y), batch_size=self.batch_size)

        return pd.DataFrame(np.concatenate(
            [self.model(batch['inputs'].to(self.device),
                        batch['targets'].to(self.device)).sigmoid().cpu().detach().numpy()
             for batch in dl]),
            index=ds[typ].X.index,
            columns=ds[typ].y.columns)

    @abstractmethod
    def extract_features(self, ds: DataManager, typ: str) -> pd.DataFrame:
        ds.active_fold = self.fold_id
        dl = torch.utils.data.DataLoader(MoADataSet(ds[typ].X, ds[typ].y), batch_size=self.batch_size)

        return pd.DataFrame(np.concatenate(
            [self.model.extract_features(batch['inputs'].to(self.device)).cpu().detach().numpy()
             for batch in dl]),
            index=ds[typ].X.index)

    @abstractmethod
    def save(self):
        torch.save(self.model.state_dict(), self.save_path)
        return self

    @abstractmethod
    def load(self):
        self.model.load_state_dict(torch.load(self.save_path))
        return self
