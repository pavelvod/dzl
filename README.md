# DZL

Package which helps to organize model runs for tabular data


Example:

```python
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.datasets import make_classification
from sklearn import metrics
from dzl import DZLClassifier, OOFValidCallback


X, y = make_classification(n_samples=5000, weights=[0.2, 0.8], n_clusters_per_class=4, n_informative=10)

y = pd.Series(y).add_prefix('item_')
X = pd.DataFrame(X, index=y.index).add_prefix('feature_')
w = pd.Series(sklearn.utils.class_weight.compute_sample_weight('balanced', y), index=y.index)
grp = X.sum(1).div(5).astype(np.int32).abs()
display(X.head(5))
display(y.head(5))


oof_clbk = OOFValidCallback()
model = DZLClassifier(model_cls=sklearn.linear_model.LogisticRegression,
                          model_params=dict(),
                          fit_params=dict(),
                          cv_params=dict(shuffle=True),
                          callbacks=[oof_clbk])

model.fit(X,y)
oof_score = metrics.roc_auc_score(y, oof_clbk.oof[:, 1])

print(f'\n\nAUC: {oof_score}')


oof_clbk = OOFValidCallback()
model = DZLClassifier(model_cls=sklearn.linear_model.LogisticRegression,
                          model_params=dict(),
                          fit_params=dict(),
                          cv_cls=sklearn.model_selection.GroupKFold,
                          cv_params=dict(),
                          callbacks=[oof_clbk],
                          
                     )

model.fit(X,y, groups=grp)
oof_score = metrics.roc_auc_score(y, oof_clbk.oof[:, 1])

print(f'\n\nAUC: {oof_score}')
```