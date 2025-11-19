import yaml

class Config:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        self._build(cfg_dict)

    def _build(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config._from_dict(v))
            else:
                setattr(self, k, v)

    @staticmethod
    def _from_dict(d: dict):
        obj = Config.__new__(Config)
        obj._build(d)
        return obj