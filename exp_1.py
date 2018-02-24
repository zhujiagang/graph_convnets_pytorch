import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import pdb #pdb.set_trace()
import collections
import time
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_my_data_sets('datasets', one_hot=False)# load data in folder datasets/
train_data = mnist.train.images.astype(np.float32)
val_data = mnist.validation.images.astype(np.float32)
test_data = mnist.test.images.astype(np.float32)
train_labels = mnist.train.labels
val_labels = mnist.validation.labels
test_labels = mnist.test.labels
print(train_data.shape)
print(train_labels.shape)
print(val_data.shape)
print(val_labels.shape)
print(test_data.shape)
print(test_labels.shape)


# class definition
class ConvNet_LeNet5(nn.Module):
    def __init__(self, net_parameters):

        print('ConvNet: LeNet5\n')

        super(ConvNet_LeNet5, self).__init__()

        Nx, Ny, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F = net_parameters
        FC1Fin = CL2_F * (Nx // 4) ** 2

        # graph CL1
        self.conv1 = nn.Conv2d(1, CL1_F, CL1_K, padding=(2, 2))
        Fin = CL1_K ** 2;
        Fout = CL1_F;
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.conv1.weight.data.uniform_(-scale, scale)
        self.conv1.bias.data.fill_(0.0)

        # graph CL2
        self.conv2 = nn.Conv2d(CL1_F, CL2_F, CL2_K, padding=(2, 2))
        Fin = CL1_F * CL2_K ** 2;
        Fout = CL2_F;
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.conv2.weight.data.uniform_(-scale, scale)
        self.conv2.bias.data.fill_(0.0)

        # FC1
        self.fc1 = nn.Linear(FC1Fin, FC1_F)
        Fin = FC1Fin;
        Fout = FC1_F;
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.fc1.weight.data.uniform_(-scale, scale)
        self.fc1.bias.data.fill_(0.0)
        self.FC1Fin = FC1Fin

        # FC2
        self.fc2 = nn.Linear(FC1_F, FC2_F)
        Fin = FC1_F;
        Fout = FC2_F;
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.fc2.weight.data.uniform_(-scale, scale)
        self.fc2.bias.data.fill_(0.0)

        # max pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, d):

        # CL1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # CL2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # FC1
        x = x.permute(0, 3, 2, 1).contiguous()  # reshape from pytorch array to tensorflow array
        x = x.view(-1, self.FC1Fin)
        x = self.fc1(x)
        x = F.relu(x)
        x = nn.Dropout(d)(x)

        # FC2
        x = self.fc2(x)

        return x

    def loss(self, y, y_target, l2_regularization):

        loss = nn.CrossEntropyLoss()(y, y_target)

        l2_loss = 0.0
        for param in self.parameters():
            data = param * param
            l2_loss += data.sum()

        loss += 0.5 * l2_regularization * l2_loss

        return loss

    def update(self, lr):

        update = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        return update

    def update_learning_rate(self, optimizer, lr):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer

    def evaluation(self, y_predicted, test_l):

        _, class_predicted = torch.max(y_predicted.data, 1)
        return 100.0 * (class_predicted == test_l).sum() / y_predicted.size(0)


# Delete existing network if exists
try:
    del net
    print('Delete existing network\n')
except NameError:
    print('No existing network to delete\n')

# network parameters
Nx = Ny = 28
CL1_F = 32
CL1_K = 5
CL2_F = 64
CL2_K = 5
FC1_F = 512
FC2_F = 10
net_parameters = [Nx, Ny, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F]

# instantiate the object net of the class
net = ConvNet_LeNet5(net_parameters)
if torch.cuda.is_available():
    net.cuda()
print(net)

# Weights
L = list(net.parameters())

# learning parameters
learning_rate = 0.05
dropout_value = 0.5
l2_regularization = 5e-4
batch_size = 100
num_epochs = 20
train_size = train_data.shape[0]
nb_iter = int(num_epochs * train_size) // batch_size
print('num_epochs=', num_epochs, ', train_size=', train_size, ', nb_iter=', nb_iter)

# Optimizer
global_lr = learning_rate
global_step = 0
decay = 0.95
decay_steps = train_size
lr = learning_rate
optimizer = net.update(lr)

# loop over epochs
indices = collections.deque()
for epoch in range(num_epochs):  # loop over the dataset multiple times

    # reshuffle
    indices.extend(np.random.permutation(train_size))  # rand permutation

    # reset time
    t_start = time.time()

    # extract batches
    running_loss = 0.0
    running_accuray = 0
    running_total = 0
    while len(indices) >= batch_size:

        # extract batches
        batch_idx = [indices.popleft() for i in range(batch_size)]
        train_x, train_y = train_data[batch_idx, :].T, train_labels[batch_idx].T
        train_x = np.reshape(train_x, [28, 28, batch_size])[:, :, :, None]
        train_x = np.transpose(train_x, [2, 3, 1, 0])  # reshape from pytorch array to tensorflow array
        train_x = Variable(torch.FloatTensor(train_x).type(dtypeFloat), requires_grad=False)
        train_y = train_y.astype(np.int64)
        train_y = torch.LongTensor(train_y).type(dtypeLong)
        train_y = Variable(train_y, requires_grad=False)

        # Forward
        y = net.forward(train_x, dropout_value)
        loss = net.loss(y, train_y, l2_regularization)
        loss_train = loss.data[0]

        # Accuracy
        acc_train = net.evaluation(y, train_y.data)

        # backward
        loss.backward()

        # Update
        global_step += batch_size  # to update learning rate
        optimizer.step()
        optimizer.zero_grad()

        # loss, accuracy
        running_loss += loss_train
        running_accuray += acc_train
        running_total += 1

        # print
        if not running_total % 100:  # print every x mini-batches
            print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (
            epoch + 1, running_total, loss_train, acc_train))

    # print
    t_stop = time.time() - t_start
    print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
          (epoch + 1, running_loss / running_total, running_accuray / running_total, t_stop, lr))

    # update learning rate
    lr = global_lr * pow(decay, float(global_step // decay_steps))
    optimizer = net.update_learning_rate(optimizer, lr)

    # Test set
    running_accuray_test = 0
    running_total_test = 0
    indices_test = collections.deque()
    indices_test.extend(range(test_data.shape[0]))
    t_start_test = time.time()
    while len(indices_test) >= batch_size:
        batch_idx_test = [indices_test.popleft() for i in range(batch_size)]
        test_x, test_y = test_data[batch_idx_test, :].T, test_labels[batch_idx_test].T
        test_x = np.reshape(test_x, [28, 28, batch_size])[:, :, :, None]
        test_x = np.transpose(test_x, [2, 3, 1, 0])  # reshape from pytorch array to tensorflow array
        test_x = Variable(torch.FloatTensor(test_x).type(dtypeFloat), requires_grad=False)
        y = net.forward(test_x, 0.0)
        test_y = test_y.astype(np.int64)
        test_y = torch.LongTensor(test_y).type(dtypeLong)
        test_y = Variable(test_y, requires_grad=False)
        acc_test = net.evaluation(y, test_y.data)
        running_accuray_test += acc_test
        running_total_test += 1
    t_stop_test = time.time() - t_start_test
    print('  accuracy(test) = %.3f %%, time= %.3f' % (running_accuray_test / running_total_test, t_stop_test))

print ("here")
print ("here")