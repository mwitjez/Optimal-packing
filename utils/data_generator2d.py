import random
from collections import defaultdict

class DataGenerator():
    def __init__(self, data_size, bin_x, bin_y):
        self.data_size = data_size
        self.bin_x = bin_x
        self.bin_y = bin_y

    def generate(self):
        data = []
        for i in range(self.data_size):
            rectangles = []
            y_cutting_points = sorted(random.sample(range(1, self.bin_y), self.bin_x // 3))
            y_positions = [0, *y_cutting_points, self.bin_y]
            for y in range(len(y_positions) - 1):
                x_cutting_points = sorted(random.sample(range(1, self.bin_x), self.bin_x // 3))
                x_positions = [0, *x_cutting_points, self.bin_x]
                for x in range(len(x_positions) - 1):
                    rectangles.append((x_positions[x + 1] - x_positions[x], y_positions[y + 1] - y_positions[y]))
            shuffled_rectangles = rectangles.copy()
            random.shuffle(shuffled_rectangles)
            index_dict = defaultdict(list)
            shuffled_rectangles_copy = shuffled_rectangles.copy()
            # Create a dictionary to store indexes of items in the shuffled list
            for item in shuffled_rectangles:
                index_dict[item].append(shuffled_rectangles_copy.index(item))
                shuffled_rectangles_copy[shuffled_rectangles_copy.index(item)] = None

            # Create a list of indexes for the shuffled list
            sequence = []
            for item in rectangles:
                sequence.append(index_dict[item][0])
                index_dict[item].pop(0)

            data.append((shuffled_rectangles, sequence, (self.bin_x, self.bin_y)))
        return data
