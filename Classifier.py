# Train a model for hand written digit recognition using:
# Date loader
# Datasets
# Batch size/ training
# Activation function

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torchvision

if __name__ == '__main__':
    print("torch with zeros:\n", torch.zeros(5,5))
    print("torch that is empty:\n ", torch.empty(5,5))
    print("torch with random numbers:\n", torch.rand(5,5))
    # device = torch.device("cuda" if torch.a.is_avalable() else "cpu")
