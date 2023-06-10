import json
import os

from collections import defaultdict
from utils.rectangle import Rectangle
from utils.cuboid import Cuboid
from utils.singleton import Singleton


class Data(metaclass=Singleton):
    def __init__(self):
        self.data_2d = self._load_2d_data()
        self.data_3d = self._load_3d_data()

    def _load_2d_data(self):
        data = defaultdict(dict)
        for file in os.listdir("data/2D"):
            if file.endswith(".json"):
                filename = file.strip(".json")
                with open(f"data/2D/{file}", 'r') as f:
                    data[filename] = json.load(f)
                data[filename]["items"] = self._create_rectangle_list(data[filename]["items"])
        return data

    def _load_3d_data(self):
        data = defaultdict(dict)
        for file in os.listdir("data/3D"):
            if file.endswith(".json"):
                filename = file.strip(".json")
                with open(f"data/3D/{file}", 'r') as f:
                    data[filename] = json.load(f)
                data[filename]["items"] = self._create_cuboid_list(data[filename]["items"])
        return data

    def _create_rectangle_list(self, data):
        return [Rectangle(rectangle['width'], rectangle['height']) for rectangle in data]

    def _create_cuboid_list(self, data):
        return [Cuboid(cuboid['width'], cuboid['height'], cuboid['depth']) for cuboid in data]
