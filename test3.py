import random

def shuffle_and_sort_indexes(lst):
    # Create a shuffled list of tuples
    shuffled_lst = lst.copy()
    random.shuffle(shuffled_lst)
    
    # Generate sort indexes to sort the shuffled list back to the original order
    sort_indexes = [i for i, _ in sorted(enumerate(lst), key=lambda x: shuffled_lst.index(x[1]))]
    
    return shuffled_lst, sort_indexes

# Example list of tuples
original_list = [(4, 2), (1, 1), (3, 2), (2, 2), (1, 1)]

shuffled_list, sort_indexes = shuffle_and_sort_indexes(original_list)

print("Original List:", original_list)
print("Shuffled List:", shuffled_list)
print("Sort Indexes:", sort_indexes)

# Sorting the shuffled list using sort_indexes
sorted_list = [shuffled_list[i] for i in sort_indexes]
print("Sorted List:", sorted_list)
