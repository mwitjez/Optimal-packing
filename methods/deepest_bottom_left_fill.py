import itertools
import numpy as np

from utils.bin3D import Bin3D


class DeepestBottomLeftPacker:
    """
    A class representing the Deepest-Bottom-Left Packing Algorithm.
    """
    def __init__(self, rectangles: list, bin_width: int, bin_height: int, bin_depth: int) -> None:
        self.rectangles = rectangles
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.bin_depth = bin_depth

    def get_max_height(self, packing_order: list):
        """Returns the maximum height of the bin after packing the rectangles in the given order."""
        packed_bin = self.pack_rectangles(packing_order)
        if packed_bin is None:
            return None
        max_height = packed_bin.map.nonzero()[2].max() + 1
        return max_height

    def get_packing_density(self, packing_order: list):
        """Calculates the packing density of a bin."""
        packed_bin = self.pack_rectangles(packing_order)
        if packed_bin is None:
            return None
        max_width = packed_bin.map.nonzero()[0].max() + 1
        max_depth = packed_bin.map.nonzero()[1].max() + 1
        max_height = packed_bin.map.nonzero()[2].max() + 1
        total_area = max_height * max_width * max_depth
        ones_area = np.sum(packed_bin.map)
        packing_density = ones_area / total_area

        return packing_density

    def pack_rectangles(self, packing_order: list):
        """Packs the rectangles in the given order using the deepest-bottom-left algorithm."""
        bin = Bin3D(self.bin_width, self.bin_height, self.bin_depth)
        sorted_rectangles = [self.rectangles[i] for i in packing_order]
        for rectangle in sorted_rectangles:
            position = self._find_valid_position(bin, rectangle)
            if position is None:
                return None
            rectangle.x, rectangle.y, rectangle.z = position
            bin.items.append(rectangle)
            self._mark_positions(bin, rectangle, *position)
        return bin

    def _find_valid_position(self, bin, rectangle):
        """Finds a valid position for the given rectangle in the bin."""
        for z in range(bin.depth - rectangle.depth + 1):
            for y in range(bin.height - rectangle.height + 1):
                for x in range(bin.width - rectangle.width + 1):
                    if self._is_valid_position(bin, rectangle, x, y, z):
                        return x, y, z
        return None

    def _is_valid_position(self, bin, rectangle, x, y, z):
        """Checks if the given position is valid for the rectangle."""
        for i, j, k in itertools.product(range(y, y + rectangle.height),
                                        range(x, x + rectangle.width),
                                        range(z, z + rectangle.depth)):
            if bin.map[i][j][k] == 1:
                return False
        return True

    def _mark_positions(self, bin, rectangle, x, y, z):
        """Marks the positions of the rectangle in the bin map."""
        bin.map[y:y+rectangle.height, x:x+rectangle.width, z:z+rectangle.depth] = 1
