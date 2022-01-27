import os

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import CrossEntropyLoss, MSELoss

from models import LeNet, AlexNet, VGGNet, GoogLeNet, ResNet50
from data import create_dataloader
from train import do_n_epochs
from utils import plot_results, save, load, write_acc

def main():
    data_name = 'MNIST'
    num_classes = None
    path_to_train_data = ''
    path_to_valid_data = ''

    model_name = 'VGGNet'
    save_model = True
    save_path = None
    load_model = False
    load_path = None

    batch_size = 8
    lr = 1e-4
    use_lr_sched = True
    epochs = 15

    train_dataloader, valid_dataloader = create_dataloader(data_name=data_name,
                                                           path_to_train_data=path_to_train_data,
                                                           path_to_valid_data=path_to_valid_data,
                                                           batch_size=batch_size,
                                                           )

    if not num_classes:
        try:
            num_classes = len(train_dataloader.dataset.targets.unique())
        except:
            num_classes = len(set(train_dataloader.dataset.targets))
    inp_channels = train_dataloader.dataset[0][0].shape[0]

    if model_name == 'LeNet':
        model = LeNet(num_classes=num_classes)
    elif model_name == 'AlexNet':
        model = AlexNet(num_classes=num_classes, inp_channels=inp_channels)
    elif model_name == 'VGGNet':
        model = VGGNet(num_classes=num_classes, inp_channels=inp_channels)
    elif model_name == 'GoogLeNet':
        model = GoogLeNet(num_classes=num_classes, inp_channels=inp_channels)
    elif model_name == 'ResNet':
        model = ResNet50(num_classes=num_classes, inp_channels=inp_channels)
    else:
        raise ValueError('Choose a valid model name.')

    optimizer = Adam(model.parameters(),
                     lr=lr,
                     )
    
    if load_model:
        model, optimizer = load(model, optimizer, load_path)

    if use_lr_sched:
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_dataloader),
            )

    loss_fcn = CrossEntropyLoss()

    state_dict, plot_data = do_n_epochs(model=model,
                                        optimizer=optimizer,
                                        loss_fcn=loss_fcn,
                                        scheduler=scheduler,
                                        train_dataloader=train_dataloader,
                                        valid_dataloader=valid_dataloader,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        )

    plot_results(data=plot_data)
    write_acc(data=plot_data['valid_accuracy'],
              model_name=model_name,
              data_name=data_name)

    if save_model:
        if save_path:
            save(state_dict, save_path)
        else:
            save(state_dict)

if __name__ == '__main__':
    main()
