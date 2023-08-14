class Box:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

class PackedBox:
    def __init__(self, x, y, z, box):
        self.x = x
        self.y = y
        self.z = z
        self.box = box

class MaxRects3DPacker:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.free_space = [(0, 0, 0, width, height, depth)]
        self.packed_boxes = []

    def find_best_fit(self, box):
        best_fit = None
        best_fit_index = -1
        for i, space in enumerate(self.free_space):
            if box.width <= space[3] and box.height <= space[4] and box.depth <= space[5]:
                if best_fit is None or space[3] * space[4] * space[5] < best_fit[3] * best_fit[4] * best_fit[5]:
                    best_fit = space
                    best_fit_index = i
        return best_fit, best_fit_index

    def pack(self, boxes):
        for box in boxes:
            best_fit, best_fit_index = self.find_best_fit(box)
            if best_fit is not None:
                x, y, z, width, height, depth = best_fit
                self.packed_boxes.append(PackedBox(x, y, z, box))
                
                if box.width == width and box.height == height and box.depth == depth:
                    del self.free_space[best_fit_index]
                else:
                    if box.width < width:
                        self.free_space.append((x + box.width, y, z, width - box.width, height, depth))
                    if box.height < height:
                        self.free_space.append((x, y + box.height, z, width, height - box.height, depth))
                    if box.depth < depth:
                        self.free_space.append((x, y, z + box.depth, width, height, depth - box.depth))

    def display_packing(self):
        for packed_box in self.packed_boxes:
            print(f"Packed Box (x:{packed_box.x}, y:{packed_box.y}, z:{packed_box.z}) - "
                  f"Dimensions (w:{packed_box.box.width}, h:{packed_box.box.height}, d:{packed_box.box.depth})")

# Example usage
box1 = Box(5, 3, 2)
box2 = Box(4, 4, 4)
box3 = Box(3, 3, 6)

packer = MaxRects3DPacker(10, 10, 10)
packer.pack([box1, box2, box3])

packer.display_packing()
