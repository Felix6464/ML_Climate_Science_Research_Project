import torch

def normalize_data(data):
    # Calculate the mean and standard deviation along the feature dimension
    mean = torch.mean(data, dim=1, keepdim=True)
    print(mean)
    std = torch.std(data, dim=1, keepdim=True)
    print(std)

    # Apply normalization using the mean and standard deviation
    normalized_data = torch.zeros_like(data)

    for i in range(len(mean)):
        normalized_data[i, :] = (data[i, :] - mean[i]) / std[i]


    return normalized_data

# Generating a random input tensor with 2 data points and 3 features
data = torch.rand((2, 3))

# Normalize the data
normalized_data = normalize_data(data)

print("Original Data:")
print(data)
print("Normalized Data:")
print(normalized_data)