from dataclasses import dataclass


@dataclass
class Cuboid:
    """
    A class representing a cuboid item to be packed.
    """
    width: int
    height: int
    depth: int
    x: int
    y: int
    z: int

    def __init__(self, width: int, height: int, depth : int):
        self.width = width
        self.height = height
        self.depth = depth
        self.x = None
        self.y = None
        self.z = None
