'''
Specifications
The first file, train.py, will train a new network on a dataset and save the model as a checkpoint.

Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu

Note:
Most of the code is reused from the Jupyter notebook portion of the project

'''

# Imports for pytorch

import matplotlib.pyplot as plt
import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

#Other imports
import json
import argparse
import time 
from collections import OrderedDict

parser = argparse.ArgumentParser(description='train.py: Trains a new network on a dataset and save the model as a checkpoint.')
parser.add_argument('--data_dir', help='Directory path to training data', required=True)
parser.add_argument('--save_dir', help='Directory path to write checkpoint data', required=False)
parser.add_argument('--arch', help='Architecture for pre-trained CNN used to detect image features', required=False)
parser.add_argument('--learning_rate', help='', required=False)
parser.add_argument('--hidden_units', help='Architecture for pre-trained CNN used to detect image features', required=False)
parser.add_argument('--epochs', help='Number of teraining epochs', required=False)
parser.add_argument('--gpu', help='Train on the gpu or the cpu', required=False)

print("This is train.py")

args = parser.parse_args()

#print(args)

################################################################################################################
###### Global variables and hyperparameters
#in actual project, data is in 'flowers' 

if args.data_dir:
	data_dir = args.data_dir
else:
	data_dir = '.'
	
if args.save_dir:
	save_dir = args.save_dir
else:
	save_dir = '.'
    
if args.arch == 'vgg19':
	arch = 'vgg19'
else:
	arch = 'vgg16'

if args.learning_rate:
	learning_rate = float(args.learning_rate)
else:
	learning_rate = 0.001
	
if args.hidden_units:
	hidden_units = int(args.hidden_units)
else:
	hidden_units = 1024
	
if args.epochs:
	epochs = int(args.epochs)
else:
	epochs = 20
	
if args.gpu and args.gpu == 'cpu':
	device = 'cpu'
else:
	device = 'cuda'
	
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

################################################################################################################	
##### Set up training transformations
resize_size = 255
pixel_size = 224

#Training transform: 
#Implements random scaling, cropping, and flipping.
#Transforms to a tensor
#Normalizes mean and standard deviation to [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
training_transform = transforms.Compose([transforms.RandomRotation((-45,45), resample=False, expand=False, center=None),
transforms.RandomResizedCrop(pixel_size, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
transforms.RandomHorizontalFlip(p=0.5),    
transforms.ToTensor(),
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#Training transform: 
#Crops to 224x224
#Transforms to a tensor
#Normalizes mean and standard deviation to [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
testing_transform = transforms.Compose([transforms.Resize(resize_size),
transforms.CenterCrop(pixel_size),
transforms.ToTensor(),
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

validation_transform = transforms.Compose([transforms.Resize(resize_size),
transforms.CenterCrop(pixel_size),
transforms.ToTensor(),
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# Load the datasets with ImageFolder
image_datasets = {}
image_datasets[train_dir] = datasets.ImageFolder(train_dir, transform=training_transform)
image_datasets[test_dir] = datasets.ImageFolder(test_dir, transform=testing_transform)
image_datasets[valid_dir] = datasets.ImageFolder(valid_dir, transform=validation_transform)

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {}
dataloaders[train_dir] = torch.utils.data.DataLoader(image_datasets[train_dir], batch_size=64, shuffle=True)
dataloaders[test_dir] = torch.utils.data.DataLoader(image_datasets[test_dir], batch_size=64, shuffle=True)
dataloaders[valid_dir] = torch.utils.data.DataLoader(image_datasets[valid_dir], batch_size=64, shuffle=True)

################################################################################################################
#### Build and train the classifier

#1. Load a pre-trained network (If you need a starting point, the VGG networks work great and are straightforward to use)
#2. Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
#3. Train the classifier layers using backpropagation using the pre-trained network to get the features
#4. Track the loss and accuracy on the validation set to determine the best hyperparameters

num_classes = 102

#1 Load a pre-trained network (as suggested, models.vgg16)
if arch == 'vgg19':
    model = models.vgg19(pretrained=True)
else:
    model = models.vgg16(pretrained=True)
#2 Define a new network

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
#Builds three hidden layers of size hidden_units, hidden_units/4
#Dropout of 0.4 after the first two
hu4 = int(hidden_units / 4)

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('drop1', nn.Dropout(p=0.40)),
                          ('fc2', nn.Linear(hidden_units, hu4)),
                          ('relu', nn.ReLU()),
                          ('drop2', nn.Dropout(p=0.40)),
                          ('fc3', nn.Linear(hu4, num_classes)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier


#3. Train the classifier layers using backpropagation using the pre-trained network to get the features
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

#epochs = 15 ->> now set by parameter
print_every = 40
steps = 0

if device == 'cuda':
    # change to cuda
    model.to('cuda')

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(dataloaders[train_dir]):
        steps += 1
        
		#moves the tensors to cpu or gpu as selected by the user
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))
            
            running_loss = 0

#4. Track the loss and accuracy on the validation set to determine the best hyperparameters

    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloaders[valid_dir]:
            images2, labels2 = data
            images = images2.to(device)
            labels = labels2.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))
	
################################################################################################################
#### Do validation on the test set

def check_accuracy_on_test(testloader):    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images2, labels2 = data
            images = images2.to(device)
            labels = labels2.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the ' + str(len(images)) +' test images: %d %%' % (100 * correct / total))

check_accuracy_on_test(dataloaders[test_dir])

################################################################################################################
#### Save the checkpoint
model.class_to_idx = image_datasets[train_dir].class_to_idx
model.cpu()
savepath = save_dir + '/chkpt.pytorchsave'
torch.save({'arch': arch, 'epoch': epochs + 1, 'state_dict': model.state_dict(), 'class_to_idx': model.class_to_idx, 'hidden_units' : hidden_units}, savepath)
