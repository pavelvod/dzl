import pathlib
import yaml


def get_object_class(txt):
    kls = txt.split('.')[-1]
    import_path = '.'.join(txt.split('.')[:-1])
    mod = __import__(import_path, fromlist=[kls])
    return getattr(mod, kls)


def load_from_config(config_path, inner_path):
    config_path = pathlib.Path(config_path)
    with config_path.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for p in inner_path.split('/'):
        config = config[p]
    return get_object_class(config['model_class'])(**config.get('model_args', {}), **config.get('other_args', {}))


class ConfigObjectManager:
    def __init__(self, config_path):
        self.config_path = pathlib.Path(config_path)

    def create(self, inner_path: str):
        return load_from_config(config_path=self.config_path, inner_path=inner_path)


if __name__ == '__main__':
    config = dict(models=dict(model=dict(model_class='sklearn.linear_model.LogisticRegression',
                                         model_args=dict(C=0.1),
                                         other_args=dict()
                                         ),
                              cv_fold=dict(model_class='sklearn.model_selection.StratifiedKFold',
                                           model_args=dict(n_splits=5, shuffle=True, random_state=42)),
                              cboost=dict(model_class='catboost.CatBoostClassifier',
                                          model_args=dict(iterations=999999,
                                                          depth=10))

                              )
                  )
    yaml.dump(config, open('config.yml', 'w'))

    conf_mgr = ConfigObjectManager('config.yml')

    print(conf_mgr.create('models/model'))
    print(conf_mgr.create('models/cv_fold'))
    print(conf_mgr.create('models/cboost'))
