from typing import Optional
import pandas as pd
from sklearn import metrics

from dzl import ModelClassifierCV
from dzl.src.callbacks import FoldMetricCallback, OOFValidCallback


class AdversarialValidator:
    def __init__(self, model_cv_params: dict, verbose: bool = True):
        self.score_fn = metrics.roc_auc_score
        self.verbose: bool = verbose
        self.model_cv_params = model_cv_params
        self.adv_model = None
        self.score = None

    def validate(self, x_trn, x_tst, callbacks: Optional[list] = None):
        X_adv = pd.concat([x_tst, x_trn]).sort_index()
        y_adv = pd.concat([pd.Series(1, index=x_trn.index), pd.Series(0, index=x_tst.index)]).sort_index()

        oof_clbk = OOFValidCallback()
        callbacks = callbacks or []
        callbacks.append(oof_clbk)
        if self.verbose:
            callbacks.append(FoldMetricCallback(metric_list=[self.score_fn]))
        self.adv_model = ModelClassifierCV(**self.model_cv_params, callbacks=callbacks)
        self.adv_model.fit(X_adv, y_adv)
        self.score = self.score_fn(y_adv, oof_clbk.oof[:, 1])
        return self
