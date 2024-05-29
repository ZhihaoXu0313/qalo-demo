import json


class NestedDict:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            return None

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)
    

class JSONParameters:
    def __init__(self, fpath="config.json") -> None:
        self.fpath = fpath
        self._params = self._load_json()
        self._parse_params(self._params)
    
    def _load_json(self):
        with open(self.fpath, 'r') as file:
            data = json.load(file)
        return data
    
    def _parse_params(self, params):
        for key, value in params.items():
            if isinstance(value, dict):
                setattr(self, key, NestedDict(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            return None
        
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self) -> str:
        return str(self._params)



    