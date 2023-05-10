import random
import json

class DataGenerator:
    def __init__(self, items_number, box_width, box_height, box_depth) -> None:
        self.items_number = items_number
        self.box_width = box_width
        self.box_height = box_height
        self.box_depth = box_depth

    def generate(self):
        items = []
        if self.items_number != round(self.items_number ** (1 / 3)) ** 3:
            raise Exception("Number of items must be a perfect cube.")
        cuts = round(self.items_number ** (1. / 3)) - 1
        depth_slices = sorted(random.sample(range(1, self.box_depth), cuts))
        depth_positions = [0, *depth_slices, self.box_depth]
        for i in range(1, len(depth_positions)):
            width_slices = sorted(random.sample(range(1, self.box_width), cuts))
            width_positions = [0, *width_slices, self.box_width]
            for j in range(1, len(width_positions)):
                height_slices = sorted(random.sample(range(1, self.box_height), cuts))
                height_positions = [0, *height_slices, self.box_height]
                for k in range(1, len(height_positions)):
                    items.append({
                        "depth": depth_positions[i] - depth_positions[i-1],
                        "height": height_positions[k] - height_positions[k-1],
                        "width": width_positions[j] - width_positions[j-1],
                    })
        self_check = sum([item["depth"] * item["height"] * item["width"] for item in items])
        if self_check != self.box_width * self.box_height * self.box_depth:
            print("Something went wrong with data generation.")
        self._to_json(items)

    def _to_json(self, items):
        data = {
            "num_items": self.items_number,
            "bin_size": [self.box_width, self.box_height, self.box_depth],
            "items": items
        }
        with open(f'P{self.items_number}.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)


dg = DataGenerator(64, 20, 20, 20)
print(dg.generate())
