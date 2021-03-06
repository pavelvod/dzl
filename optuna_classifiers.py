import pathlib
from functools import partial

import namegenerator
import optuna

from base import CVTrainer
from fold_classifiers import LightGBMClassifierFoldTrainer


class OptunaOptimizer:
    def __init__(self, name, num_trials):
        self.name = name
        self.num_trials = num_trials
        self.study = optuna.create_study(direction="maximize",
                                         study_name=self.name,
                                         storage=f'sqlite:///{self.name}.sqlite3',
                                         load_if_exists=True
                                         )

    def optimize(self, ds):
        num_success_trials = len([trial.state == optuna.trial.TrialState.COMPLETE for trial in self.study.trials])
        num_remain_trials = self.num_trials - num_success_trials
        partial_func = partial(self.objective, ds=ds)
        self.study.optimize(partial_func, n_trials=num_remain_trials, gc_after_trial=True, show_progress_bar=True)

    @property
    def best_params(self):
        return self.study.best_params

    def objective(self, trial, ds):
        return 0

    def retrain(self, ds):
        pass


class LGBMOptunaOptimizer(OptunaOptimizer):
    def retrain(self, ds):
        params = dict(
            init_params={
                'objective': 'binary',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'metric': 'auc',
                'n_estimators': 999999,
            },
            fit_params=dict(verbose=-1,
                            early_stopping_rounds=100
                            )
        )

        params['init_params'].update(self.best_params)

        print(params)
        trainer = CVTrainer(fold_trainer_cls=LightGBMClassifierFoldTrainer,
                            ds=ds,
                            model_name=self.name,
                            params=params,
                            save_path=pathlib.Path('.')
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
                'metric': 'auc',
                'n_estimators': 200,
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                'max_depth': trial.suggest_int('max_depth', 2, 50),
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

        trainer = CVTrainer(fold_trainer_cls=LightGBMClassifierFoldTrainer,
                            ds=ds,
                            model_name=f'{self.name}_{namegenerator.gen()}',
                            params=params,
                            save_path=pathlib.Path('.')
                            )
        return trainer.fit_single_fold()
