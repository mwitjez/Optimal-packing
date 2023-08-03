import torch

# Create a PyTorch tensor
tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Determine the range of indices you want to shuffle
start_index = 2
end_index = 6

# Extract the desired sub-tensor that you want to shuffle
sub_tensor = tensor[start_index:end_index + 1]

# Generate a random permutation of indices
permuted_indices = torch.randperm(sub_tensor.size(0))

# Shuffle the sub-tensor using the permuted indices
shuffled_sub_tensor = sub_tensor[permuted_indices]

# Update the original tensor with the shuffled sub-tensor
tensor[start_index:end_index + 1] = shuffled_sub_tensor

print(tensor)
