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
        try:
            packed_bin = self.pack_rectangles(packing_order)
        except:
            return float('inf')
        max_height = packed_bin.map.nonzero()[0].max() + 1
        return max_height

    def pack_rectangles(self, packing_order: list):
        """Packs the rectangles in the given order using the deepest-bottom-left algorithm."""
        bin = Bin3D(self.bin_width, self.bin_height, self.bin_depth)
        sorted_rectangles = [self.rectangles[i] for i in packing_order]
        for rectangle in sorted_rectangles:
            for z in range(bin.depth - rectangle.depth + 1):
                for y in range(bin.height - rectangle.height + 1):
                    for x in range(bin.width - rectangle.width + 1):
                        if self._is_valid_position(bin, rectangle, x, y, z):
                            rectangle.x = x
                            rectangle.y = y
                            rectangle.z = z
                            bin.items.append(rectangle)
                            self._mark_positions(bin, rectangle, x, y, z)
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                raise Exception("No valid position found for rectangle.")
        return bin

    def _is_valid_position(self, bin, rectangle, x, y, z):
        """Checks if the given position is valid for the rectangle."""
        for i in range(y, y + rectangle.height):
            for j in range(x, x + rectangle.width):
                for k in range(z, z + rectangle.depth):
                    if bin.map[i][j][k] == 1:
                        return False
        return True

    def _mark_positions(self, bin, rectangle, x, y, z):
        """Marks the positions of the rectangle in the bin map."""
        bin.map[y:y+rectangle.height, x:x+rectangle.width, z:z+rectangle.depth] = 1
