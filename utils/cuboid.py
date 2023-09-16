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

    def __init__(self, width: int, height: int, depth : int, x: int = None, y: int = None, z: int = None) -> None:
        self.width = width
        self.height = height
        self.depth = depth
        self.x = x
        self.y = y
        self.z = z
