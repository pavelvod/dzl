import pathlib
from functools import partial

import namegenerator
import optuna
import torch

from base import CVTrainer
from fold_classifiers import LightGBMClassifierFoldTrainer, CatboostClassifierFoldTrainer, TabNetFoldTrainer, \
    XGBoostClassifierFoldTrainer, SklearnClassifierFoldTrainer


class OptunaOptimizer:
    def __init__(self, name, num_trials, metric='roc_auc_score'):
        self.name = name
        self.num_trials = num_trials
        self.study = optuna.create_study(direction="maximize",
                                         study_name=self.name,
                                         storage=f'sqlite:///{self.name}.sqlite3',
                                         load_if_exists=True
                                         )
        self.metric = metric

    def optimize(self, ds):
        num_success_trials = len([trial.state == optuna.trial.TrialState.COMPLETE for trial in self.study.trials])
        num_remain_trials = self.num_trials - num_success_trials
        partial_func = partial(self.objective, ds=ds)
        self.study.optimize(partial_func, n_trials=num_remain_trials, gc_after_trial=True, show_progress_bar=True)
        return self.retrain(ds)

    @property
    def best_params(self):
        return self.study.best_params

    def objective(self, trial, ds):
        return 0

    def retrain(self, ds):
        pass


class LGBMOptunaOptimizer(OptunaOptimizer):
    def __init__(self, name, num_trials, min_depth=1, max_depth=99, *args, **kwargs):
        super().__init__(name=name, num_trials=num_trials, *args, **kwargs)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def retrain(self, ds):
        params = dict(
            init_params={
                'objective': 'binary',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                # 'metric': 'auc',
                'n_estimators': 999999,
                'learning_rate': 0.03
            },
            fit_params=dict(verbose=-1,
                            early_stopping_rounds=100
                            )
        )

        if self.metric == 'roc_auc_score':
            params['init_params']['metric'] = 'auc'

        params['init_params'].update(self.best_params)

        print(params)
        trainer = CVTrainer(fold_trainer_cls=LightGBMClassifierFoldTrainer,
                            ds=ds,
                            model_name=self.name,
                            params=params,
                            save_path=pathlib.Path('../..'),
                            metric=self.metric
                            )
        trainer.fit()
        trainer.save()
        print(trainer.score('trn'), trainer.score('val'))

    def objective(self, trial, ds):
        params = dict(
            init_params={
                'objective': 'binary',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                # 'metric': 'auc',
                'n_estimators': 99999999,
                'learning_rate': 0.03,
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1E-16, 25.0),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1E-16, 25.0),
                'reg_alpha': trial.suggest_loguniform('lambda_l2', 1E-16, 25.0),
                'reg_lambda': trial.suggest_loguniform('lambda_l2', 1E-16, 25.0),
                'subsample': trial.suggest_float('subsample ', 1E-16, 1.0),
                'cat_smooth': trial.suggest_float('cat_smooth', 1.0, 50.0),
                'max_depth': trial.suggest_int('max_depth', self.min_depth, self.max_depth),
                'num_leaves': trial.suggest_int('num_leaves', 2, 1024 * 32),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 64),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 3000),
            },
            fit_params=dict(verbose=-1,
                            early_stopping_rounds=100
                            )
        )

        if self.metric == 'roc_auc_score':
            params['init_params']['metric'] = 'auc'

        trainer = CVTrainer(fold_trainer_cls=LightGBMClassifierFoldTrainer,
                            ds=ds,
                            model_name=f'{self.name}_{namegenerator.gen()}',
                            params=params,
                            save_path=pathlib.Path('./cache')
                            )
        return trainer.fit_single_fold()


class CatBoostOptunaOptimizer(OptunaOptimizer):
    def __init__(self, name, num_trials, min_depth=1, max_depth=99, *args, **kwargs):
        super().__init__(name=name, num_trials=num_trials, *args, **kwargs)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def retrain(self, ds):
        params = dict(
            init_params={
                'iterations': 999999,
                # 'eval_metric': 'AUC',
                'verbose': 100,
                'early_stopping_rounds': 100,
                'learning_rate': 0.1
            },
            fit_params=dict()
        )

        if self.metric == 'roc_auc_score':
            params['init_params']['eval_metric'] = 'AUC'

        params['init_params'].update(self.best_params)

        print(params)
        trainer = CVTrainer(fold_trainer_cls=CatboostClassifierFoldTrainer,
                            ds=ds,
                            model_name=self.name,
                            params=params,
                            save_path=pathlib.Path('../..'),
                            metric=self.metric
                            )
        trainer.fit()
        trainer.save()
        print(trainer.score('trn'), trainer.score('val'))

    def objective(self, trial, ds):
        init_params = {
            'iterations': 500,
            # 'eval_metric': 'AUC',
            'verbose': 100,
            'early_stopping_rounds': 100,
            'learning_rate': 0.1,
            'rsm': 0.1,
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 100),
            "depth": trial.suggest_int("depth", self.min_depth, self.max_depth),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
        }

        if self.metric == 'roc_auc_score':
            init_params['eval_metric'] = 'AUC'

        if init_params["bootstrap_type"] == "Bayesian":
            init_params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif init_params["bootstrap_type"] == "Bernoulli":
            init_params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        params = dict(
            init_params=init_params,
            fit_params=dict()
        )

        print(params)
        trainer = CVTrainer(fold_trainer_cls=CatboostClassifierFoldTrainer,
                            ds=ds,
                            model_name=f'{self.name}_{namegenerator.gen()}',
                            params=params,
                            save_path=pathlib.Path('./cache')
                            )
        return trainer.fit_single_fold()


class XGBoostOptunaOptimizer(OptunaOptimizer):
    def __init__(self, name, num_trials, min_depth=1, max_depth=99, *args, **kwargs):
        super().__init__(name=name, num_trials=num_trials, *args, **kwargs)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def retrain(self, ds):
        init_params = {

            'n_estimators': 99999999,
            'learning_rate': 0.01
        }

        if self.metric == 'roc_auc_score':
            init_params['eval_metric'] = 'auc'

        params = dict(
            init_params=init_params,
            fit_params=dict(verbose=100)
        )

        params['init_params'].update(self.best_params)

        print(params)
        trainer = CVTrainer(fold_trainer_cls=XGBoostClassifierFoldTrainer,
                            ds=ds,
                            model_name=self.name,
                            params=params,
                            save_path=pathlib.Path('../..'),
                            metric=self.metric
                            )
        trainer.fit()
        trainer.save()
        print(trainer.score('trn'), trainer.score('val'))
        return trainer

    def objective(self, trial, ds):
        init_params = {
            'n_estimators': 500,
            # 'eval_metric': 'auc',
            'learning_rate': 0.01,

            "booster": trial.suggest_categorical("booster", ["gbtree"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        }

        if self.metric == 'roc_auc_score':
            init_params['eval_metric'] = 'auc'

        if init_params["booster"] == "gbtree" or init_params["booster"] == "dart":
            init_params["max_depth"] = trial.suggest_int("max_depth", self.min_depth, self.max_depth)
            init_params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            init_params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            init_params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if init_params["booster"] == "dart":
            init_params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            init_params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            init_params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            init_params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        params = dict(
            init_params=init_params,
            fit_params=dict(verbose=100)
        )

        print(params)
        trainer = CVTrainer(fold_trainer_cls=XGBoostClassifierFoldTrainer,
                            ds=ds,
                            model_name=f'{self.name}_{namegenerator.gen()}',
                            params=params,
                            save_path=pathlib.Path('./cache'),
                            metric=self.metric
                            )
        return trainer.fit_single_fold()


class TabNetOptunaOptimizer(OptunaOptimizer):
    def __init__(self, name, num_trials, min_steps=1, max_steps=99, *args, **kwargs):
        super().__init__(name=name, num_trials=num_trials, *args, **kwargs)
        self.min_steps = min_steps
        self.max_steps = max_steps

    def retrain(self, ds):
        init_params = dict(n_d=self.best_params['n_da'],
                           n_a=self.best_params['n_da'],
                           optimizer_fn=torch.optim.Adam,
                           optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                           scheduler_params=dict(mode="min",
                                                 patience=2,
                                                 min_lr=1e-5,
                                                 factor=0.5, ),
                           scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau
                           )

        init_params.update(self.best_params)
        params = dict(
            init_params=init_params,
            fit_params=dict(),
            config=dict(pretrain=True)
        )
        del params['init_params']['n_da']
        print(params)
        trainer = CVTrainer(fold_trainer_cls=TabNetFoldTrainer,
                            ds=ds,
                            model_name=self.name,
                            params=params,
                            save_path=pathlib.Path('../..'),
                            metric=self.metric
                            )
        trainer.fit()
        trainer.save()
        print(trainer.score('trn'), trainer.score('val'))

    def objective(self, trial, ds):
        n_da = trial.suggest_int("n_da", 4, 64, step=4)
        init_params = dict(n_d=n_da,
                           n_a=n_da,
                           n_steps=trial.suggest_int("n_steps", self.min_steps, self.max_steps, step=1),
                           gamma=trial.suggest_float("gamma", 1., 1.4, step=0.2),
                           lambda_sparse=trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
                           mask_type=trial.suggest_categorical("mask_type", ["entmax", "sparsemax"]),
                           n_shared=trial.suggest_int("n_shared", 1, 3),
                           optimizer_fn=torch.optim.Adam,
                           optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                           scheduler_params=dict(mode="min",
                                                 patience=2,
                                                 min_lr=1e-5,
                                                 factor=0.5, ),
                           scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                           )

        if self.metric == 'roc_auc_score':
            eval_metric = 'auc'

        params = dict(
            init_params=init_params,
            fit_params=dict(max_epochs=5, eval_metric=['eval_metric']),
            config=dict(pretrain=False)
        )

        trainer = CVTrainer(fold_trainer_cls=TabNetFoldTrainer,
                            ds=ds,
                            model_name=f'{self.name}_{namegenerator.gen()}',
                            params=params,
                            save_path=pathlib.Path('./cache')
                            )

        return trainer.fit_single_fold()


class HistGradientBoostingOptunaOptimizer(OptunaOptimizer):
    def __init__(self, name, num_trials, *args, **kwargs):
        super().__init__(name=name, num_trials=num_trials, *args, **kwargs)

    def retrain(self, ds):
        init_params = dict(max_iter=9999,
                           validation_fraction=0.1,
                           n_iter_no_change=100)
        init_params.update(self.best_params)

        params = {'pipeline': {'sklearn.ensemble.HistGradientBoostingClassifier': init_params}}

        print(params)
        trainer = CVTrainer(fold_trainer_cls=SklearnClassifierFoldTrainer,
                            ds=ds,
                            model_name=self.name,
                            params=params,
                            save_path=pathlib.Path('../..'),
                            metric=self.metric
                            )
        trainer.fit()
        trainer.save()
        print(trainer.score('trn'), trainer.score('val'))

    def objective(self, trial, ds):
        init_params = dict(max_iter=9999,
                           validation_fraction=0.1,
                           n_iter_no_change=100,
                           learning_rate=trial.suggest_float('learning_rate', 1e-6, 1.0),
                           max_depth=trial.suggest_int('max_depth', 1, 16),
                           min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 1024 * 32),
                           max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 1, 1024 * 32),
                           max_bins=trial.suggest_int('max_bins', 1, 255)
                           )

        params = {'pipeline': {'sklearn.ensemble.HistGradientBoostingClassifier': init_params}}
        print(params)
        trainer = CVTrainer(fold_trainer_cls=SklearnClassifierFoldTrainer,
                            ds=ds,
                            model_name=f'{self.name}_{namegenerator.gen()}',
                            params=params,
                            save_path=pathlib.Path('./cache')
                            )

        return trainer.fit_single_fold()


class LogRegOptunaOptimizer(OptunaOptimizer):
    def __init__(self, name, num_trials, *args, **kwargs):
        super().__init__(name=name, num_trials=num_trials, *args, **kwargs)

    def retrain(self, ds):
        init_params = dict()
        init_params.update(self.best_params)

        params = dict(pipeline={

            'sklearn.linear_model.LogisticRegression': init_params
        }
        )

        print(params)
        trainer = CVTrainer(fold_trainer_cls=SklearnClassifierFoldTrainer,
                            ds=ds,
                            model_name=self.name,
                            params=params,
                            save_path=pathlib.Path('../..'),
                            metric=self.metric
                            )
        trainer.fit()
        trainer.save()
        print(trainer.score('trn'), trainer.score('val'))

    def objective(self, trial, ds):
        init_params = dict(C=trial.suggest_float('C', 1e-4, 25.))

        params = dict(pipeline={
            'sklearn.linear_model.LogisticRegression': init_params
        }
        )

        print(params)
        trainer = CVTrainer(fold_trainer_cls=SklearnClassifierFoldTrainer,
                            ds=ds,
                            model_name=f'{self.name}_{namegenerator.gen()}',
                            params=params,
                            save_path=pathlib.Path('./cache'),
                            metric='log_loss'
                            )

        return trainer.fit_single_fold()
