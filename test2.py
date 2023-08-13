from rectpack import newPacker
from rectpack.maxrects import MaxRects

import numpy as np

rectangles = [(2,2),(1,2),(5,5)]
bins = [(10, 10)]

packer = newPacker(sort_algo=None)

# Add the rectangles to packing queue
for r in rectangles:
	packer.add_rect(*map(int, r))

# Add the bins where the rectangles will be placed
for b in bins:
	packer.add_bin(*b)

# Start packing
packer.pack()

map = np.zeros((10, 10))

# Draw the packed rectangles on the map
for rect in packer[0]:
	map[rect.corner_bot_l.x:rect.corner_top_r.x, rect.corner_bot_l.y:rect.corner_top_r.y] = 1

max_height = map.nonzero()[0].max() + 1
total_area = map.shape[0] * max_height
ones_area = np.sum(map)
packing_density = ones_area / total_area
print("Packing density: ", packing_density)
np.set_printoptions(threshold=np.inf)

# Print the map as a NumPy array
print(map)
