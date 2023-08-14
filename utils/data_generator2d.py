import random


class DataGenerator:
    MIN_BIN_SIZE = 10

    def __init__(self, data_size, bin_max_x: int = 10, bin_max_y: int = 10):
        self.data_size = data_size
        if bin_max_x < self.MIN_BIN_SIZE or self.MIN_BIN_SIZE < 10:
            raise ValueError("Bin size must be at least 10x10")
        self.bin_max_x = bin_max_x
        self.bin_max_y = bin_max_y

    def generate(self):
        data = []
        for i in range(self.data_size):
            bin_x = random.randint(self.MIN_BIN_SIZE, self.bin_max_x)
            bin_y = random.randint(self.MIN_BIN_SIZE, self.bin_max_y)
            rectangles = []
            y_cutting_points = sorted(random.sample(range(1, bin_y), bin_x // 3))
            y_positions = [0, *y_cutting_points, bin_y]
            for y in range(len(y_positions) - 1):
                x_cutting_points = sorted(
                    random.sample(range(1, bin_x), bin_x // 3)
                )
                x_positions = [0, *x_cutting_points, bin_x]
                for x in range(len(x_positions) - 1):
                    rectangles.append(
                        (
                            x_positions[x + 1] - x_positions[x],
                            y_positions[y + 1] - y_positions[y],
                        )
                    )
            shuffled_rectangles = rectangles.copy()
            random.shuffle(shuffled_rectangles)
            # TODO do funkcji może się przydać później
            # index_dict = defaultdict(list)
            # shuffled_rectangles_copy = shuffled_rectangles.copy()
            # for item in shuffled_rectangles:
            #     index_dict[item].append(shuffled_rectangles_copy.index(item))
            #     shuffled_rectangles_copy[shuffled_rectangles_copy.index(item)] = None
            # sequence = []
            # for item in rectangles:
            #     sequence.append(index_dict[item][0])
            #     index_dict[item].pop(0)

            data.append((shuffled_rectangles, (bin_x, bin_y)))
        return data
