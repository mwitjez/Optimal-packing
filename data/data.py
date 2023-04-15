import json

from utils.rectangle import Rectangle

class Data:
    def __init__(self):
        self.C1 = self._load('data/C1.json')
        self.C2 = self._load('data/C2.json')
        self.C3 = self._load('data/C3.json')

    def _load(self, file):
        with open(file, 'r') as f:
            data = json.load(f)
        data["items"] = self.create_rectangle_list(data["items"])
        return data

    def create_rectangle_list(self, data):
        return [Rectangle(rectangle['width'], rectangle['height']) for rectangle in data]
