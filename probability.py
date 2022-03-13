import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
import torchvision
import numpy as np
from scipy.stats import multivariate_normal


def normal_p(X, input):
    x_reshape = X.reshape(X.shape[0], -1).T
    input_reshape = input.reshape(input.shape[0], -1)
    mean_X = x_reshape.mean(axis=1)
    std_X = x_reshape.std(axis=1)
    std_X[std_X < 0.1] = 0.1
    rv = multivariate_normal(mean_X, std_X, allow_singular=True)
    pX = rv.pdf(input_reshape)
    pX = np.sqrt(pX)
    pX /= pX.sum()
    return pX


def entropy(p, q):
    p_flatten = p.flatten()
    q_flatten = q.flatten()
    return -np.dot(p_flatten, np.log(q_flatten))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x

    def forwards_output(self, x):
        z1 = self.conv1(x)
        z2 = self.conv2(z1)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        z2 = z2.view(z2.size(0), -1)
        z3 = self.out(z2)
        return z1, z2, z3


class Probability:
    def __init__(self, mnist_train, net):
        self.Y_size = 10
        self.mnist_train = mnist_train
        self.net = net

        self.I_xz1 = []
        self.I_xz2 = []
        self.I_xz3 = []

        self.I_yz1 = []
        self.I_yz2 = []
        self.I_yz3 = []

    def get_pY(self):
        self.pY = np.ones(self.Y_size) * (1. / self.Y_size)

    def get_pX(self):
        self.pX = normal_p(self.X, self.X)

    def get_pX_y(self):
        self.pX_y = np.zeros((self.X.shape[0], 10))
        for y_label in range(10):
            x_current = self.X[self.Y == y_label]
            pX_y = normal_p(x_current, self.X)
            self.pX_y[:, y_label] = pX_y

    def get_pXY(self):
        self.pXY = self.pX_y * 0.1

    def get_pZ(self):
        self.pZ1 = normal_p(self.Z1, self.Z1)
        self.pZ2 = normal_p(self.Z2, self.Z2)
        self.pZ3 = normal_p(self.Z3, self.Z3)

    def get_pZ_y(self):
        self.pZ1_y = np.zeros((self.Z1.shape[0], 10))
        for y_label in range(10):
            z_current = self.Z1[self.Y == y_label]
            pZ_y = normal_p(z_current, self.Z1)
            self.pZ1_y[:, y_label] = pZ_y

        self.pZ2_y = np.zeros((self.Z2.shape[0], 10))
        for y_label in range(10):
            z_current = self.Z2[self.Y == y_label]
            pZ_y = normal_p(z_current, self.Z2)
            self.pZ2_y[:, y_label] = pZ_y

        self.pZ3_y = np.zeros((self.Z3.shape[0], 10))
        for y_label in range(10):
            z_current = self.Z3[self.Y == y_label]
            pZ_y = normal_p(z_current, self.Z3)
            self.pZ3_y[:, y_label] = pZ_y

    def get_pZY(self):
        self.p_Z1Y = self.pZ1_y * 0.1
        self.p_Z2Y = self.pZ2_y * 0.1
        self.p_Z3Y = self.pZ3_y * 0.1


    def get_PXZ(self):
        self.p_Z1_x = np.ones((self.X.shape[0], self.X.shape[0]))
        self.p_Z1_x += (np.eye(self.X.shape[0]) * 4)
        self.p_Z1_x = self.p_Z1_x / self.p_Z1_x.sum()
        self.p_Z2_x = self.p_Z1_x
        self.p_Z3_x = self.p_Z1_x

    def get_Inf(self):
        x_entropy = entropy(self.pX, self.pX)
        y_entropy = entropy(self.pY, self.pY)
        z1_entropy = entropy(self.pZ1, self.pZ1)
        z2_entropy = entropy(self.pZ2, self.pZ2)
        z3_entropy = entropy(self.pZ3, self.pZ3)

        yz1_entropy = entropy(self.p_Z1Y, self.p_Z1Y)
        yz2_entropy = entropy(self.p_Z2Y, self.p_Z2Y)
        yz3_entropy = entropy(self.p_Z3Y, self.p_Z3Y)


        Iyz1 = y_entropy + z1_entropy - yz1_entropy
        Iyz2 = y_entropy + z2_entropy - yz2_entropy
        Iyz3 = y_entropy + z3_entropy - yz3_entropy

        self.I_xz1.append(z1_entropy)
        self.I_xz2.append(z2_entropy)
        self.I_xz3.append(z3_entropy)

        self.I_yz1.append(Iyz1)
        self.I_yz2.append(Iyz2)
        self.I_yz3.append(Iyz3)


    def get_X_Y_Z(self):
        X_array = []
        Y_array = []
        Z1_array = []
        Z2_array = []
        Z3_array = []
        for i, (images, labels) in enumerate(self.mnist_train):
            X_array.append(images.cpu().detach().numpy())
            Y_array.append(labels.cpu().detach().numpy())
            z1, z2, z3 = self.net.forwards_output(images)
            Z1_array.append(z1.cpu().detach().numpy())
            Z2_array.append(z2.cpu().detach().numpy())
            Z3_array.append(z3.cpu().detach().numpy())
        self.X = np.concatenate(X_array)
        self.Y = np.concatenate(Y_array)
        self.Z1 = np.concatenate(Z1_array)
        self.Z2 = np.concatenate(Z2_array)
        self.Z3 = np.concatenate(Z3_array)

    def get_Z(self):
        Z1_array = []
        Z2_array = []
        Z3_array = []
        for i, (images, labels) in enumerate(self.mnist_train):
            z1, z2, z3 = self.net.forwards_output(images)
            Z1_array.append(z1.cpu().detach().numpy())
            Z2_array.append(z2.cpu().detach().numpy())
            Z3_array.append(z3.cpu().detach().numpy())
        self.Z1 = np.concatenate(Z1_array)
        self.Z2 = np.concatenate(Z2_array)
        self.Z3 = np.concatenate(Z3_array)

    def step(self, is_first=True):
        if is_first:
            self.get_X_Y_Z()
            self.get_pX()
            print("------get pX-------")
            self.get_pY()
            self.get_pX_y()
            self.get_pXY()
            print("------get pXY------")
        else:
            self.get_Z()
        print("-------get pZ--------")
        self.get_pZ()
        self.get_pZ_y()
        self.get_pZY()
        print("-------get all prob-------")
        self.get_Inf()
        print("-------get entropy--------")


def train(net, loss_func, optimizer, mnist_train):
    net.train()
    num_epochs = 10
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(mnist_train):
            output = net(images)[0]
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                torch.save(net.state_dict(), "mnist_model.pth")

def test(net, mnist_test):
    net.eval()
    for i, (images, labels) in enumerate(mnist_test):
        test_output, last_layer = net(images)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        print("output: ", pred_y)
        print("labels: ", labels)
        print("test_output_last: ", test_output[-1])


if __name__ == '__main__':
    net = CNN()
    train_data = datasets.MNIST(
        root = 'data',
        train = True,
        transform = ToTensor(),
        download = True,
    )

    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )

    mnist_train = torch.utils.data.DataLoader(train_data,
                                              batch_size=100,
                                              shuffle=True,
                                              num_workers=1)

    mnist_test = torch.utils.data.DataLoader(test_data,
                                             batch_size=100,
                                             shuffle=True,
                                             num_workers=1)

    total_step = len(mnist_train)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    #train(net, loss_func, optimizer, mnist_train)
    net.load_state_dict(torch.load("mnist_model.pth"))
    #test(net, mnist_test)

    probality = Probability(mnist_train, net)
    probality.get_X_Y_Z()
    probality.get_pX()
    probality.get_pY()
    probality.get_pX_y()
    probality.get_pXY()
    probality.get_pZ()
    #probality.get_pZ_y()
    probality.get_PXZ()
    probality.get_IXZ()

