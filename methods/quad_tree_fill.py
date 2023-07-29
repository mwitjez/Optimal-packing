import numpy as np

from scipy.spatial import cKDTree
from utils.bin import Bin


class QuadTreePacker:
    """
    A class representing the Bottom-Left Packing Algorithm.
    """
    def __init__(self, rectangles: list, bin_width: int, bin_height: int) -> None:
        self.rectangles = rectangles
        self.bin_width = bin_width
        self.bin_height = bin_height

    def get_max_height(self, packing_order: list):
        """Returns the maximum height of the bin after packing the rectangles in the given order."""
        packed_bin = self.pack_rectangles(packing_order)
        if packed_bin is None:
            return None
        max_height = packed_bin.map.nonzero()[0].max() + 1
        return max_height

    def get_packing_density(self, packing_order: list):
        """Calculates the packing density of a bin."""
        packed_bin = self.pack_rectangles(packing_order)
        if packed_bin is None:
            return None
        max_height = packed_bin.map.nonzero()[0].max() + 1
        total_area = packed_bin.map.shape[0] * max_height
        ones_area = np.sum(packed_bin.map)
        packing_density = ones_area / total_area

        return packing_density

    def pack_rectangles(self, packing_order: list):
        """Packs the rectangles in the given order."""
        bin = Bin(self.bin_width, self.bin_height)
        sorted_rectangles = [self.rectangles[i] for i in packing_order]
        positions = [(x, y) for y in range(bin.height) for x in range(bin.width)]
        tree = cKDTree(positions)
        for rectangle in sorted_rectangles:
            valid_positions = []
            for x, y in positions:
                if self._is_valid_position(bin, rectangle, x, y):
                    valid_positions.append((x, y))
            if valid_positions:
                _, indices = tree.query(valid_positions)
                x, y = positions[indices[0]]
                rectangle.x = x
                rectangle.y = y
                bin.items.append(rectangle)
                self._mark_positions(bin, rectangle, x, y)
            else:
                return None
        return bin

    def _is_valid_position(self, bin, rectangle, x, y):
        """Returns True if the rectangle can be placed at the given position, False otherwise."""
        height, width = bin.map.shape
        if y + rectangle.height > height or x + rectangle.width > width:
            return False
        for i in range(y, y + rectangle.height):
            for j in range(x, x + rectangle.width):
                if bin.map[i][j] == 1:
                    return False
        return True

    def _mark_positions(self, bin, rectangle, x, y):
        """Marks the positions of the rectangle in the bin map."""
        bin.map[y:y+rectangle.height, x:x+rectangle.width] = 1
