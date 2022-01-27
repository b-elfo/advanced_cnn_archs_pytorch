import json

import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def save(state_dict: dict,
         path: str = 'model_checkpoint',
         ):
    torch.save(state_dict,path)

def load(model: nn.Module,
         optimizer: torch.optim,
         path: str,
         ):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optim_state_dict'])
    return model, optimizer

def plot_results(data: dict,
              ):
    fig, axs = plt.subplots(2,1,figsize=(30,20))
    y1 = data['train_loss']
    y2 = data['valid_loss']
    y3 = data['train_accuracy']
    y4 = data['valid_accuracy']
    assert len(y1)==len(y2)
    assert len(y3)==len(y4)
    num_epochs = len(y1)
    x = range(num_epochs)
    axs[0].plot(x, y1, label='train loss')
    axs[0].plot(x, y2, label='valid loss')
    axs[0].set_title('Train vs Valid Loss')
    axs[0].legend()
    axs[1].plot(x, y3, label='train accuracy')
    axs[1].plot(x, y4, label='valid accuracy')
    axs[1].set_title('Train vs Valid Accuracy')
    axs[1].legend()
    plt.show()

def write_acc(data: list,
              model_name: str,
              data_name: str,
              ):
    fn = data_name+'_valid_acc_results.txt'
    with open(fn, 'a+') as file:
        file.write('{}\n{}\n'.format(model_name, str(data)))
        file.close()