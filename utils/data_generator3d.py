import random


class DataGenerator:
    MIN_BIN_SIZE = 5
    ADDITIONAL_HEIGHT = 10

    def __init__(self, data_size, bin_max_x: int = 10, bin_max_y: int = 10, bin_max_z: int = 10):
        self.data_size = data_size
        if bin_max_x < self.MIN_BIN_SIZE or bin_max_y < self.MIN_BIN_SIZE or bin_max_z < self.MIN_BIN_SIZE:
            raise ValueError("Bin size must be at least 10x10x10")
        self.bin_max_x = bin_max_x
        self.bin_max_y = bin_max_y
        self.bin_max_z = bin_max_z

    def generate(self):
        data = []
        for i in range(self.data_size):
            bin_x = random.randint(self.MIN_BIN_SIZE, self.bin_max_x)
            bin_y = random.randint(self.MIN_BIN_SIZE, self.bin_max_y)
            bin_z = random.randint(self.MIN_BIN_SIZE, self.bin_max_z)
            cuboids = []
            z_cutting_points = sorted(
                random.sample(range(1, bin_z), bin_z // 3)
            )
            z_positions = [0, *z_cutting_points, bin_z]
            for z in range(len(z_positions) - 1):
                y_cutting_points = sorted(random.sample(range(1, bin_y), bin_y // 3))
                y_positions = [0, *y_cutting_points, bin_y]
                for y in range(len(y_positions) - 1):
                    x_cutting_points = sorted(
                        random.sample(range(1, bin_x), bin_x // 3)
                    )
                    x_positions = [0, *x_cutting_points, bin_x]
                    for x in range(len(x_positions) - 1):
                            cuboids.append(
                                (
                                    x_positions[x + 1] - x_positions[x],
                                    y_positions[y + 1] - y_positions[y],
                                    z_positions[z + 1] - z_positions[z],
                                )
                            )
            random.shuffle(cuboids)
            data.append((cuboids, (bin_x, bin_y, bin_z+self.ADDITIONAL_HEIGHT)))
        return data
