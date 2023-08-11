import random
from collections import defaultdict

original_list = [(4, 2), (1, 1), (3, 2), (2, 2), (1, 1)]
# Shuffle the original list
shuffled_list = original_list.copy()
random.shuffle(shuffled_list)
index_dict = defaultdict(list)
shuffled_list_copy = shuffled_list.copy()

# Create a dictionary to store indexes of items in the shuffled list
for item in shuffled_list:
    index_dict[item].append(shuffled_list_copy.index(item))
    shuffled_list_copy[shuffled_list_copy.index(item)] = None

# Create a list of indexes for the shuffled list
indexes = []
for item in original_list:
    indexes.append(index_dict[item][0])
    index_dict[item].pop(0)

sorted_list = [shuffled_list[i] for i in indexes]

# Compare the sorted list to the original list
if sorted_list == original_list:
    print("The shuffled list is the same as the original list.")
else:
    print("The shuffled list is different from the original list.")

print("Original List:", original_list)
print("Shuffled List:", shuffled_list)
print("Sorted List:", sorted_list)
print("Indexes:", indexes)