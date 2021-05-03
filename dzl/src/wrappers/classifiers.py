from .base import BaseCVClassifierWrapper


class LGBMCVClassifierWrapper(BaseCVClassifierWrapper):

    def _fit(self, model, x_trn, y_trn, x_val, y_val, *args, **kwargs):
        model.fit(x_trn, y_trn,
                  eval_set=[(x_trn, y_trn), (x_val, y_val)],
                  *args, **kwargs)
        return self
