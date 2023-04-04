import json

from utils.rectangle import Rectangle

class Data:
    def __init__(self):
        self._c1 = self._load('data/C1.json')
        self.P1_rectangles = self.create_rectangle_list(self._c1['P1'])

    def _load(self, file):
        with open(file, 'r') as f:
            return json.load(f)
 
    def create_rectangle_list(self, data):
        return [Rectangle(rectangle['w'], rectangle['h']) for rectangle in data]
