import pathlib
from dataclasses import field, dataclass
import yaml


@dataclass
class ModelConfig:
    model_cls: type
    init_params: dict = field(default_factory=dict)
    fit_params: dict = field(default_factory=dict)
    predict_params: dict = field(default_factory=dict)
    transform_params: dict = field(default_factory=dict)

    def to_yaml(self, path: pathlib.Path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)

        return self

    @classmethod
    def from_yaml(self, path: pathlib.Path):
        with open(path, 'r') as f:
            d = yaml.load(f)
        return ModelConfig(**d)
