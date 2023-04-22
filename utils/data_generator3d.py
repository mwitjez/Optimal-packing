import random
import json

import random
import json

def generate_dataset(num_items, bin_size):
    items = []
    cuts = num_items - 1
    for i in range(num_items):
        if i == num_items - 1:
            cut = bin_size
        else:
            if random.choice([True, False]):
                cut = [bin_size[0], bin_size[1], random.randint(1, bin_size[2] - 1)]
            else:
                cut = [bin_size[0], random.randint(1, bin_size[1] - 1), bin_size[2]]
        items.append({"depth": cut[0], "height": cut[1], "width": cut[2]})
        bin_size = [bin_size[j] - cut[j] for j in range(3)]
        if i < num_items - 1:
            num_cuts = min(cuts, random.randint(1, num_items - i))
            for j in range(num_cuts):
                if random.choice([True, False]):
                    cut = [bin_size[0], bin_size[1], random.randint(1, bin_size[2] - 1)]
                else:
                    cut = [bin_size[0], random.randint(1, bin_size[1] - 1), bin_size[2]]
                items.append({"depth": cut[0], "height": cut[1], "width": cut[2]})
                bin_size = [bin_size[k] - cut[k] for k in range(3)]
            cuts -= num_cuts
    dataset = {"num_items": num_items, "bin_size": [bin_size[0], bin_size[1], bin_size[2]], "items": items}
    return json.dumps(dataset, indent=4)

# Example usage
num_items = 9
bin_size = [10, 10, 20]
dataset = generate_dataset(num_items, bin_size)
print(dataset)
