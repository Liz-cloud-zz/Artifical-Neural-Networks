# Train a model for hand written digit recognition using:
# Date loader
# Datasets
# Batch size/ training
# Activation function

# use gpu operator to fasten up our operators
import torch
# used for neural networks for data loader
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
# transforms to transform images
from torchvision import transforms
from torch import optim
import os
import sys
script_location = os.path.split(os.path.realpath(sys.argv[0]))[0]

if __name__ == '__main__':

    # used to put all data into gpu and model include gpu and  we can train all of these together inside the gpu itself

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # hyper - parameters declared and set
    num_epochs=10
    num_classes=10
    gamma=0.001
    batch_size=64
    input_size=784 # 28 x 28
    hidden_layer=100 # hidden neurons
    # script_location = os.path.split(os.path.realpath(sys.argv[0]))[0]
    # print(script_location)

    train_data = torchvision.datasets.MNIST(root="./",train=True,transform=transforms.ToTensor(), download=False)
    test_data=torchvision.datasets.MNIST(root="./",train=False,transform=transforms.ToTensor(),download=False)

    train_loader=torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True, num_workers=2)
    test_loader=torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False, num_workers=2)

    # check the data samples
    check_data= iter(train_loader)
    image,label=next(check_data)
    print(image.shape,label.shape)

    # try to manipulate and view the images
    for i in range(1,60+1):
        plt.subplot(6,10,i)
        plt.imshow(image[0].numpy().squeeze(),cmap='gray_r')
