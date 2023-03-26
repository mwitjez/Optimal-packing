import json


class Data:
    def __init__(self):
        self.c1 = self._load('data/C1.json')

    def _load(self, file):
        with open(file, 'r') as f:
            return json.load(f)
