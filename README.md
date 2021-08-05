# DZL

Package which helps to organize model runs for tabular data


Example:

```python

import pandas as pd
from dzl import ModelClassifierCV, OOFValidCallback
import sklearn.linear_model
from sklearn.datasets import make_classification
from sklearn import metrics

X, y = make_classification(n_samples=5000)

y = pd.Series(y).add_prefix('item_')
X = pd.DataFrame(X, index=y.index).add_prefix('feature_')


display(X.head(5))
display(y.head(5))


oof_clbk = OOFValidCallback()
model = ModelClassifierCV(model_cls=sklearn.linear_model.LogisticRegression,
                          model_params=dict(),
                          fit_params=dict(),
                          cv_params=dict(shuffle=True),
                          callbacks=[oof_clbk])

model.fit(X,y)
oof_score = metrics.roc_auc_score(y, oof_clbk.oof[:, 1])

print(f'\n\nAUC: {oof_score}')
```