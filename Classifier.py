# Train a model for hand written digit recognition using:
# Date loader
# Datasets
# Batch size/ training
# Activation function

# use gpu operator to fasten up our operators
import numpy as np
import torch
# used for neural networks for data loader
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
# transforms to transform images
from torchvision import transforms
from torch import optim
from time import time


# import os
# import sys

# script_location = os.path.split(os.path.realpath(sys.argv[0]))[0]


def main():
    # used to put all data into gpu and model include gpu
    # and  we can train all of these together inside the gpu itself
    # when you have huge amount of data use a gpu system so you can train your data faster
    # you can get the reserves faster to analyze stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # hyper - parameters declared and set
    # change num of epochs , gamma, batch size to give better output
    num_epochs = 10
    num_classes = 10
    gamma = 0.001
    batch_size = 64
    input_size = 784  # 28 x 28
    hidden_layer = 100  # hidden neurons
    # script_location = os.path.split(os.path.realpath(sys.argv[0]))[0]
    # print(script_location)

    train_data = torchvision.datasets.MNIST(root="./", train=True, transform=transforms.ToTensor(), download=False)
    test_data = torchvision.datasets.MNIST(root="./", train=False, transform=transforms.ToTensor(), download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # check the data samples
    check_data = iter(train_loader)
    image, label = next(check_data)
    print(image.shape, label.shape)

    # try to manipulate and view the images
    for i in range(1, 60 + 1):
        print(label[i])
        plt.subplot(6, 10, i)
        plt.imshow(image[0].numpy().squeeze(), cmap='gray_r')
    plt.show()

    # implement model for Neural Network
    # increase the number of  Layers for deep learning/network
    # use more hidden layers to reduce the loss
    # using one hidden layer makes a shallow neural network
    model = nn.Sequential(
        nn.Linear(in_features=input_size, out_features=hidden_layer),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layer, out_features=hidden_layer),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layer, out_features=hidden_layer),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layer, out_features=hidden_layer),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layer, out_features=num_classes)
    )
    repr(model)
    # the loss using the cross Entropy loss because when it is multi class classification
    # but if it was binary class we did use Binary CLass entropy loss
    criterion = nn.CrossEntropyLoss()
    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.Adam(model.parameters(), lr=gamma)

    samples = len(train_loader)  # len total samples

    # put model inside of gpu
    #
    model = model.to(device)

    # make training loop COre of Neural Network
    time0 = time()
    for ep in range(num_epochs):
        for step, (image, label) in enumerate(train_loader):
            # pass images to device
            image = image.reshape(-1, 784).to(device)
            # pass labels to device
            label = label.to(device)

            # get output
            output = model(image)
            # get loss
            loss = criterion(output, label)

            # Clear the gradients, do this because gradients are accumulated, Training pass:
            optimizer.zero_grad()
            # go back using back propagation by zeroing the gradients, then update weights
            # This is where the model learns using backpropagation:
            loss.backward()
            # Take an update step and few the new weights i.e. Optimize the weights:
            optimizer.step()

            print("Epoch-> {}/{}, Step-> {}/{}, Loss-> {:.4f}".format(ep, num_epochs, step, samples, loss.item()))

            # if training losses are decreases that is a good thing
    print("\nTime taken to train is (in minutes)-> ", (time() - time0) / 60)

    # view images with their classes:
    # use the test dataset
    image, label = next(iter(test_loader))
    img=image[0].view(1, 784)

    # Turn off gradients to speed up this part
    with torch.no_grad():
        log_probs = model(img.cuda())

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(log_probs)
    probability = list(ps.cpu().numpy()[0])
    print("Predicted Digit ->", probability.index(max(probability)))
    viewClassification(img.view(1, 28, 28), ps)


def viewClassification(image, ps):
    # Function for viewing an image and its predicted classes.

    ps = ps.cpu().data.numpy().squeeze()
    figure, (axis1, axis2) = plt.subplot(figsize=(6, 9), nclos=2)
    axis1.imshow(image.resize_(1, 28, 28).numpy().squeeze())
    axis2.barh(np.arange(10), ps)
    axis2.set_aspect(0.1)
    axis2.set_yticks(np.arange(10))
    axis2.set_yticklabels(np.arange(10))
    axis2.set_title('Class Probability')
    axis2.set_xlim(0, 1.1)
    plt.tight_layout()


if __name__ == '__main__':
    main()
