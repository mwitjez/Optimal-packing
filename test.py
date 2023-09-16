class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.x = 0
        self.y = 0

def goodness_number(rect, horizontal_line, vertical_line):
    gn = 0
    if rect.width == horizontal_line:
        gn += 1
    if rect.height == vertical_line:
        gn += 1
    return gn

def lwf_packing(rectangles, container_width, container_height):
    packed_rectangles = []
    remaining_space = container_width * container_height

    for rect in rectangles:
        best_gn = -1
        best_position = (0, 0)

        for x in range(container_width - rect.width + 1):
            for y in range(container_height - rect.height + 1):
                if is_position_valid(rect, x, y, packed_rectangles):
                    gn = goodness_number(rect, x, container_height - y)
                    if gn > best_gn:
                        best_gn = gn
                        best_position = (x, y)

        if best_gn > -1:
            rect.x, rect.y = best_position
            packed_rectangles.append(rect)
            remaining_space -= rect.width * rect.height

    return packed_rectangles, remaining_space

def is_position_valid(rect, x, y, packed_rectangles):
    for packed_rect in packed_rectangles:
        if not (x + rect.width <= packed_rect.x or x >= packed_rect.x + packed_rect.width or
                y + rect.height <= packed_rect.y or y >= packed_rect.y + packed_rect.height):
            return False
    return True

# Sample usage
rectangles = [Rectangle(2, 3), Rectangle(4, 5), Rectangle(1, 6)]
container_width = 10
container_height = 10
packed_rectangles, remaining_space = lwf_packing(rectangles, container_width, container_height)
print(f"Packed {len(packed_rectangles)} rectangles with {remaining_space} units of space remaining.")
