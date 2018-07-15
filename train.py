#!/usr/bin/env python3
"""
@author: Stefan Filipiak
"""

import argparse
import os
from collections import OrderedDict
from time import time, strftime, gmtime

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models

def main():
    start_time = time()

    in_arg = get_input_args()
    device = torch.device("cuda:0" if in_arg.gpu and torch.cuda.is_available() else "cpu")
    
    data = load_data(in_arg.data_dir)
    model, input_size = load_arch(in_arg.arch)
    
    criterion, optimizer = create_classifier(in_arg.hidden_units, in_arg.output_units, in_arg.learning_rate, model, input_size)
    network_worker('train', data, model, criterion, optimizer, in_arg.epochs, device, 50)
    network_worker('test', data, model, criterion, optimizer, in_arg.epochs, device, 5)
    
    save_checkpoint(in_arg.save_dir, model, data, optimizer, in_arg.epochs, in_arg.arch)
    
    tot_time = strftime("%H:%M:%S", gmtime(time() - start_time))
    print("\n** Total Elapsed Runtime:", tot_time)
    

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
    Parameters:
      None - simply using argparse module to create & store command line arguments
    Returns:
      parse_args() -data structure that stores the command line arguments object  
    """

    parser = argparse.ArgumentParser(description='Image classifier.')
    
    parser.add_argument('data_dir', type=str, help='Path to the image files. Expected subdirectories are ./train, ./valid, ./test')
    parser.add_argument('--save_dir', type=str, default='.', help='Path to where the checkpoint is stored to')
    parser.add_argument('--arch', type=str, default='vgg16', help='CNN model architecture to use for image classification. Pick any of the following vgg16, alexnet, resnet18, densenet121')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the CNN model')
    parser.add_argument('--hidden_units', type=str, default='512', help='Sizes for hidden layers. Seperate by comma if more then one')
    parser.add_argument('--output_units', type=int, default=102, help='Output size of the network')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to run')
    parser.add_argument('--gpu', type=bool, default=False, const=True, nargs='?', help='train on gpu')

    return parser.parse_args()


def load_data(data_dir):
    """
    Use torchvision to load the data. 
    Parameters:
      data_dir - Path to the image files. Expected subdirectories are ./train, ./valid, ./test
    Returns:
      parse_args() -data structure that stores the command line arguments object  
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    for directory in [train_dir, valid_dir, test_dir]:
        if not os.path.isdir(directory):
            raise IOError("Directory " + directory + " does not exist")
    
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])
    
    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=data_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)
    
    return {
        'datasets': {
            'train': train_datasets,
            'valid': valid_datasets,
            'test': test_datasets
        },
        'loader': {
            'train': trainloader,
            'valid': validloader,
            'test': testloader
        }
    }


def load_arch(arch):
    """
    Load a pretrained network
    """
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = 512
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    else:
        raise ValueError('Please choose one of \'vgg16\', \'alexnet\', \'resnet18\' or , \'densenet121\' for parameter arch.')
        
    for param in model.parameters():
        param.requires_grad = False
    
    return model, input_size
        
    
def create_classifier(hidden_units, output_units, learning_rate, model, input_size, drop_p=0.5):
    hidden_units = [int(x) for x in hidden_units.split(',')]
    hidden_units.append(output_units)
    
    if input_size < hidden_units[0]:
        raise ValueError('Please choose a hidden_unit lower than ' + str(input_size))
    
    # Add the first layer, input to a hidden layer
    hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_units[0])])
    
    # Add a variable number of more hidden layers
    layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
    hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
    
    net = OrderedDict()
    
    for i in range(len(hidden_layers)):
        net.update({'fc{}'.format(i): hidden_layers[i]})
        
        if i+1 < len(hidden_layers):
            net.update({'relu{}'.format(i): nn.ReLU()})
            net.update({'dropout{}'.format(i): nn.Dropout(p=drop_p)})
    
    net.update({'output': nn.LogSoftmax(dim=1)})
    
    model.classifier = nn.Sequential(net)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return criterion, optimizer


def network_worker(mode, data, model, criterion, optimizer, epochs, device, print_every):
    """
    Train or validate the network
    """
    model.to(device)
    
    if mode == 'train':
        loader = data['loader']['train']
    elif mode == 'test':
        loader = data['loader']['test']
    else:
        raise ValueError('Please choose \'train\' or \'test\' for parameter mode.')
    
    if mode == 'train':
        print("Start of training")    
    else:
        print("Start of validation")
        
    for epoch in range(epochs):
        model.train()
        
        running_loss = 0
        steps = 0
        start = time()
        
        for inputs, labels in iter(loader):
            steps += 1
            
            optimizer.zero_grad()
            
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)        
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                if mode == 'train':
                    print("Epoch: {}/{} ({})... ".format(epoch+1, epochs, steps),
                          "Training Loss: {:.4f}...".format(running_loss/print_every),
                          "Device: {}...Time: {:.3f}s".format(device, (time() - start)/3))
                else: # 'test' --> validate
                    validation_loss, accuracy = validate_network(data, model, criterion, device)
                    
                    print("Epoch: {}/{} ({})... ".format(epoch+1, epochs, steps),
                          "Testing Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(data['loader']['valid'])),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(data['loader']['valid'])),
                          "Device: {}...Time: {:.3f}s".format(device, (time() - start)/3))
                
                running_loss = 0
                start = time()
            
    if mode == 'train':
        print("End of training")    
    else:
        print("End of validation")


def validate_network(data, model, criterion, device):
    model.eval()
    
    accuracy = 0
    validation_loss = 0
    
    for inputs, labels in iter(data['loader']['valid']):
        inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model.forward(inputs)
        
        validation_loss += criterion(outputs, labels)
        probabilities = torch.exp(outputs).data
        equality = (labels.data == probabilities.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        
    model.train()
    
    return validation_loss, accuracy


def save_checkpoint(save_dir, model, data, optimizer, epochs, arch):
    checkpoint = {
        'model': model,
        'state_dict': model.state_dict(),
        'epochs': epochs+1,
        'optimizer': optimizer,
        'class_to_idx': data['datasets']['train'].class_to_idx
    }
    
    torch.save(checkpoint, '{}/checkpoint-{}.pth'.format(save_dir, arch))
    
    
if __name__ == '__main__':
    main()
