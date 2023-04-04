from utils.bin import Bin


class BottomLeftPacker:
    """
    A class representing the Bottom-Left Packing Algorithm.
    """
    def __init__(self, rectangles: list, bin_width: int, bin_height: int) -> None:
        self.rectangles = rectangles
        self.bin_width = bin_width
        self.bin_height = bin_height

    def get_max_height(self, packing_order):
        """Returns the maximum height of the bin after packing the rectangles in the given order."""
        packed_bin = self.pack_rectangles(packing_order)
        max_height = 0
        for item in packed_bin.items:
            current_height = item.y + item.height
            if current_height > max_height:
                max_height = current_height
        return max_height

    def pack_rectangles(self, packing_order):
        """Packs a list of Rectangle objects into a Bin object using the Bottom-Left Packing Algorithm."""
        bin = Bin(self.bin_width, self.bin_height)
        self.rectangles = [self.rectangles[i] for i in packing_order]
        for rectangle in self.rectangles:
            best_position = None
            best_y = float("inf")
            if not bin.items:
                self._handle_empty_bin(self.bin_width, self.bin_height, bin, rectangle)
            else:
                for item in bin.items:
                    if self._can_place_to_the_right(self.bin_width, rectangle, best_y, item):
                        best_position = (item.x + item.width, item.y)
                        best_y = item.y
                    elif self._can_place_below(self.bin_height, rectangle, best_y, item):
                        best_position = (item.x, item.y + item.height)
                        best_y = item.y + item.height
                    else:
                        raise Exception("The rectangle does not fit into the bin.")

                if best_position is not None:
                    rectangle.x, rectangle.y = best_position
                    bin.items.append(rectangle)

        return bin

    def _handle_empty_bin(self, bin_width, bin_height, bin, rectangle):
        """Handles the case when the bin is empty."""
        if self._can_fit(bin_width, bin_height, rectangle):
            rectangle.x = 0
            rectangle.y = 0
            bin.items.append(rectangle)
        else:
            raise Exception("The rectangle does not fit into the bin.")

    def _can_fit(self, bin_width, bin_height, rectangle):
        """Checks if a rectangle can fit into a bin."""
        return rectangle.width <= bin_width and rectangle.height <= bin_height

    def _can_place_to_the_right(self, bin_width, rectangle, best_y, item):
        """Checks if a rectangle can be placed to the right of another rectangle."""
        return item.x + item.width + rectangle.width <= bin_width and item.y <= best_y

    def _can_place_below(self, bin_height, rectangle, best_y, item):
        """Checks if a rectangle can be placed below another rectangle."""
        return item.y + item.height + rectangle.height <= bin_height and item.y + item.height <= best_y
