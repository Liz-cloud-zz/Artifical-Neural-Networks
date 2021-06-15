# Train a model for hand written digit recognition using:
# Date loader
# Datasets
# Batch size/ training
# Activation function

import torch
import torch.nn as nn
import  torchvision
import matplotlib.pyplot as plt
import torchvision.tranforms as transforms

device = torch.device("cuda" if torch.a.is_avalable() else "cpu")
