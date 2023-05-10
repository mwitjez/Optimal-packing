import json

from utils.rectangle import Rectangle
from utils.cuboid import Cuboid

class Data:
    def __init__(self):
        self.C1 = self._load_2d_data('data/2D/C1.json')
        self.C2 = self._load_2d_data('data/2D/C2.json')
        self.C3 = self._load_2d_data('data/2D/C3.json')
        self.C4 = self._load_2d_data('data/2D/C4.json')
        self.P8 = self._load_3d_data('data/3D/P8.json')
        self.P27 = self._load_3d_data('data/3D/P27.json')
        self.P64 = self._load_3d_data('data/3D/P64.json')


    def _load_2d_data(self, file):
        with open(file, 'r') as f:
            data = json.load(f)
        data["items"] = self._create_rectangle_list(data["items"])
        return data

    def _load_3d_data(self, file):
        with open(file, 'r') as f:
            data = json.load(f)
        data["items"] = self._create_cuboid_list(data["items"])
        return data

    def _create_rectangle_list(self, data):
        return [Rectangle(rectangle['width'], rectangle['height']) for rectangle in data]

    def _create_cuboid_list(self, data):
        return [Cuboid(cuboid['width'], cuboid['height'], cuboid['depth']) for cuboid in data]
