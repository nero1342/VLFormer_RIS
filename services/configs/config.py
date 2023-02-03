import yaml

class Config(object):
    def __init__(self, data):
        self.data = data
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Config(value) if isinstance(value, dict) else value

    def __getattr__(self, item):
            return None

    def __str__(self):
        return yaml.dump(self.data)