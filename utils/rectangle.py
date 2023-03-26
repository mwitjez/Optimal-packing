from dataclasses import dataclass


@dataclass
class Rectangle:
    """
    A class representing a rectangular item to be packed.
    """
    width: int
    height: int
    x: int
    y: int

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.x = None
        self.y = None
