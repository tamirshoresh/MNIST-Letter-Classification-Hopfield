# general modules:
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import os
import seaborn as sns
import sys
import torch
import sklearn
from sklearn.metrics import confusion_matrix
import cv2
import random
# for EMNIST importation:
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
# Import Hopfield-specific modules:
from hflayers import Hopfield, HopfieldPooling
# Import auxiliary modules.
from distutils.version import LooseVersion
from typing import Optional, Tuple
# Importing PyTorch specific modules.
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, LeakyReLU, Sequential, Softmax, Flatten, BatchNorm1d
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import AdamW, RMSprop
from torch.utils.data import DataLoader, random_split
from torch.backends import cudnn
from torchinfo import summary
# extras
# import tensorflow as tf
# from emnist import list_datasets, extract_training_samples, extract_test_samples
# from sklearn.preprocessing import OneHotEncoder
# from tensorflow.keras import layers

########################################################################################################################
# import EMNIST dataset:
# the EMNIST letter dataset is guven through torch datasets library
########################################################################################################################
train_and_val_data = datasets.EMNIST(
    split='letters',
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), ),
                                  transforms.ToTensor()])
)

test_data = datasets.EMNIST(
    split='letters',
    root="data",
    train=False,
    download=True,
    transform=ToTensor()#,
    #target_transform=Lambda(lambda y: torch.zeros(27, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')

# Devide train to train and validation
dataset_size = len(train_and_val_data)
val_split = 0.2
val_size = int(val_split * dataset_size)
train_size = dataset_size - val_size
training_data, val_data = random_split(train_and_val_data, [train_size, val_size])

# Create data loader of training and test set (later we need to add validation)
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, shuffle=True)
########################################################################################################################
# Creating training mechanism:
# Before digging into Hopfield-based networks, a few auxiliary variables and functions need to be defined.
# This is nothing special with respect to Hopfield-based networks, but rather common preparation work of (almost) every
# machine learning setting (e.g. definition of a data loader as well as a training loop).
########################################################################################################################
def train_epoch(network: Module,
                optimiser: AdamW,
                data_loader: DataLoader,
                loss_fn
                ) -> Tuple[float, float]:
    """
    Execute one training epoch.

    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader: data loader instance providing training data
    :return: tuple comprising training loss as well as accuracy
    """
    network.train()
    losses, accuracies = [], []
    for batch, (data, target) in enumerate(data_loader):
        data, target = data.to(device=device), target.to(device=device)
        b_size = data.size()[0] # size of batch

        optimiser.zero_grad()
        pred = network.forward(input=torch.flatten(data, start_dim=2))
        pred = torch.reshape(pred, (b_size,-1))
        loss = loss_fn(pred, target.long())
        #loss = loss_fn(torch.unsqueeze(pred, -2), target.long())

        # Backpropagation
        loss.backward()
        clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimiser.step()

        # Calculate accuracy
        curr_acc = 0
        for b in range(b_size):
            curr_acc = curr_acc + (1 if target[b] == torch.argmax(pred[b]) else 0)
        accuracy = curr_acc/b_size
        #accuracy = (1 if target == torch.argmax(pred) else 0)
        accuracies.append(accuracy)
        losses.append(loss.detach().item())


        size = len(train_dataloader.dataset)
        if batch % 80 == 0:
            loss, current = loss.item(), batch * len(data)
            print(f"loss: {loss:>7f}    accuracy: {accuracy:>7f}    [{current:>5d}/{size:>5d}]")

    return (sum(losses) / len(losses), sum(accuracies) / len(accuracies))

############################################################

def eval_iter(network: Module,
              data_loader: DataLoader,
              loss_fn
              ) -> Tuple[float, float]:
    """
    Evaluate the current model.

    :param network: network instance to evaluate
    :param data_loader: data loader instance providing validation data
    :return: tuple comprising validation loss as well as accuracy
    """
    network.eval()
    with torch.no_grad():
        losses, accuracies = [], []
        for data, target in data_loader:
            data, target = data.to(device=device), target.to(device=device)

            # Process data by Hopfield-based network.
            model_output = network.forward(input=torch.flatten(data, start_dim=-2))
            loss = loss_fn(torch.unsqueeze(model_output, -2), target.long())

            # Compute performance measures of current model.
            accuracy = (1 if target == torch.argmax(model_output) else 0)
            accuracies.append(accuracy)
            losses.append(loss.detach().item())

        # Report progress of validation procedure.
        return (sum(losses) / len(losses), sum(accuracies) / len(accuracies))

############################################################

def eval_test_set(network: Module,
                  data_loader: DataLoader,
                  loss_fn
                  ):# -> Tuple[float, float]:
    """
    Evaluate the test set.

    :param network: network instance to evaluate
    :param data_loader: data loader instance providing validation data
    :return: tuple comprising validation loss as well as accuracy
    """
    network.eval()
    with torch.no_grad():
        preds, targs = [], []
        for data, target in data_loader:
            data, target = data.to(device=device), target.to(device=device)

            # Process data by Hopfield-based network.
            model_output = network.forward(input=torch.flatten(data, start_dim=-2))

            # Fill preds and targs lists
            targs.append(target[0])
            preds.append(torch.argmax(model_output))

        # Report progress of validation procedure.
        return targs, preds


############################################################

def operate(network: Module,
            optimiser: AdamW,
            data_loader_train: DataLoader,
            data_loader_eval: DataLoader,
            loss_fn,
            num_epochs: int = 1
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train the specified network by gradient descent using backpropagation.

    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader_train: data loader instance providing training data
    :param data_loader_eval: data loader instance providing validation data
    :param num_epochs: amount of epochs to train
    :return: data frame comprising training as well as evaluation performance
    """
    losses, accuracies = {r'train': [], r'eval': []}, {r'train': [], r'eval': []}
    for epoch in range(num_epochs):
        print('\n\nStarting Epoch '+str(epoch+1))
        # Train network.
        performance = train_epoch(network, optimiser, data_loader_train, loss_fn)
        losses[r'train'].append(performance[0])
        accuracies[r'train'].append(performance[1])

        # Evaluate current model.
        performance = eval_iter(network, data_loader_eval, loss_fn)
        losses[r'eval'].append(performance[0])
        accuracies[r'eval'].append(performance[1])

    # Report progress of training and validation procedures.
    return pd.DataFrame(losses), pd.DataFrame(accuracies)

############################################################

def set_seed(seed: int = 42) -> None:
    """
    Set seed for all underlying (pseudo) random number sources.

    :param seed: seed to be used
    :return: None
    """
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

############################################################

def plot_performance(loss: pd.DataFrame,
                     accuracy: pd.DataFrame,
                     log_folder: str
                     ) -> None:
    """
    Plot and save loss and accuracy.

    :param loss: loss to be plotted
    :param accuracy: accuracy to be plotted
    :param log_file: target file for storing the resulting plot
    :return: None
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))

    loss_plot = sns.lineplot(data=loss, ax=ax[0])
    loss_plot.set(xlabel=r'Epoch', ylabel=r'Cross-entropy Loss')

    accuracy_plot = sns.lineplot(data=accuracy, ax=ax[1])
    accuracy_plot.set(xlabel=r'Epoch', ylabel=r'Accuracy')

    ax[1].yaxis.set_label_position(r'right')
    fig.tight_layout()
    plots_pdf = log_folder + r'/'
    fig.savefig(log_folder)
    #plt.show(fig)

########################################################################################################################
# Creating the Hopfield network:
# The instantiation of the heart of a Hopfield-based network, the module Hopfield, is rather straightforward.
# Only one argument, the size of the input, needs to be set. An additional output projection is defined, to down-project
# the result of Hopfield to the correct output size. Afterwards, everything is wrapped into a container of type
# torch.nn.Sequential and a corresponding optimiser is defined. After this, the Hopfield-based network and all
# auxiliaries are set up and ready to associate.
########################################################################################################################
global loss_fn
loss_fn = torch.nn.CrossEntropyLoss()

# Optimization Options:
beta_range = [3.6] # [0.036, 0.36] # 0.036 is 1/sqrt(d), where d is the length of the memory
num_heads = [1] # add 1 and 2 # number of hidden hopfield layers
lr_range = [1e-4] # learning rate
FF_size = [3000] # feed forward layer size
opts = ['rmsprop']
#dropout_rate = [0.2,0.5,0.8]


# Loop over all 21 options, train and evaluate the model
for b in beta_range:

    n_heads = num_heads[0]
    ff = FF_size[0]
    opt = opts[0]
    lr=lr_range[0]
    set_seed()
    hopfield = Hopfield(input_size=784, output_size=784, scaling=b, num_heads=n_heads)
    mid_layer = Linear(in_features=hopfield.output_size, out_features=ff)
    output_projection = Linear(in_features=ff, out_features=27)
    network = Sequential(hopfield, Flatten(), BatchNorm1d(784),
                         mid_layer, BatchNorm1d(ff), LeakyReLU(),
                         output_projection,
                         Flatten(start_dim=0)).to(device=device)
    if (opt=='adamw'):
        optimiser = AdamW(params=network.parameters(), lr=lr)
    elif(opt=='rmsprop'):
        optimiser = RMSprop(params=network.parameters(), lr=lr)

    summary(network, input_size=(1, 1, 784))

    losses, accuracies = operate(
        network=network,
        optimiser=optimiser,
        data_loader_train=train_dataloader,
        data_loader_eval=val_dataloader,
        loss_fn=loss_fn,
        num_epochs=350,
    )

    # Evaluate test set and plot confusion matrix
    targs, preds = eval_test_set(network, test_dataloader, loss_fn)
    classes = (
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z")
    cf_matrix = confusion_matrix(targs, preds)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])

    test_performance = eval_iter(network, test_dataloader, loss_fn)
    test_loss = test_performance[0]
    test_acc = test_performance[1]

    plt.figure(figsize=(15, 15))
    sn.heatmap(df_cm, annot=True, cbar=False, linewidth=.5, cmap="crest", fmt='g', annot_kws={'rotation': 45})
    plt.title('Test Set Accuracy: ' + str("%.2f" % (test_acc * 100)) + '% \nTest Set Loss: ' + str(
        "%.4f" % test_loss), fontsize=22)
    plt.ylabel("True Label", fontsize=20)
    plt.xlabel("Predicted Label", fontsize=20)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    log_path = r'C:\Users\project20\RESULTS\350 epochs\beta=3.6, lr=0.0001, parallel_heads=1, opt=rmsprop, ff=3000, dropout=0'
    os.mkdir(log_path)
    plt.savefig(log_path + '/CF.png')
    # save model & results
    torch.save(network, log_path + '/model.pth')
    plot_performance(loss=losses, accuracy=accuracies, log_folder=log_path + '/results.pdf')





