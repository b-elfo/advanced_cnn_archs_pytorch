import os

import numpy as np

import torch
from torch.nn import Module
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision


def do_epoch(model: Module,
             optimizer: optim,
             loss_fcn: Module,
             dataloader: DataLoader,
             scheduler: optim.lr_scheduler,
             ):
    model.train()

    total_loss = 0
    train_accuracy = 0
    for batch_idx, (image, target) in enumerate(dataloader):
        
        image = image.cuda()
        
        target = target.cuda()

        predict = model(image).cuda()

        loss = loss_fcn(predict, target)
        total_loss += loss

        train_accuracy += torch.sum(predict.max(1).indices == target)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if scheduler:
            scheduler.step()

    total_loss /= len(dataloader)
    train_accuracy = train_accuracy.float() / len(dataloader)
    return total_loss, train_accuracy

def do_valid_epoch(model: Module,
             loss_fcn: Module,
             dataloader: DataLoader,
             ):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        valid_accuracy = 0
        for batch_idx, (image, target) in enumerate(dataloader):
        
            image = image.cuda()
        
            target = target.cuda()

            predict = model(image).cuda()

            loss = loss_fcn(predict, target)
            total_loss += loss
            valid_accuracy += torch.sum(predict.max(1).indices == target)

        total_loss /= len(dataloader)
        valid_accuracy = valid_accuracy.float() / len(dataloader)
    return total_loss, valid_accuracy

def do_n_epochs(model: Module,
                optimizer: torch.optim,
                loss_fcn: Module,
                scheduler: optim.lr_scheduler,
                train_dataloader: DataLoader,
                valid_dataloader: DataLoader,
                epochs: int = 10,
                batch_size: int = 8,
                ):
    epoch_plot_data = {'train_loss':[], 'train_accuracy':[], 'valid_loss':[], 'valid_accuracy':[]}
    for epoch in range(epochs):
        train_loss, train_accuracy = do_epoch(model=model,
                                              optimizer=optimizer,
                                              loss_fcn=loss_fcn,
                                              dataloader=train_dataloader,
                                              scheduler=scheduler
                                              )
        valid_loss, valid_accuracy = do_valid_epoch(model=model,
                                                    loss_fcn=loss_fcn,
                                                    dataloader=valid_dataloader,
                                                    )

        epoch_plot_data['train_loss'].append(train_loss.cpu().detach().numpy())
        epoch_plot_data['train_accuracy'].append(train_accuracy.cpu().detach().numpy()/batch_size)
        epoch_plot_data['valid_loss'].append(valid_loss.cpu().detach().numpy())
        epoch_plot_data['valid_accuracy'].append(valid_accuracy.cpu().detach().numpy()/batch_size)
        print("Epoch: {}\tTrain Loss: {:.5f}\tValid Loss: {:.5f}".format(epoch, train_loss, valid_loss))

        # Solved
        if valid_accuracy/batch_size > 0.99:
            print("Solved...")
            break

    state_dict = {'model_state_dict': model.state_dict(),
                  'optim_state_dict': optimizer.state_dict(),
                  }
    return state_dict, epoch_plot_data

