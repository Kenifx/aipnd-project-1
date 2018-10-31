'''
Specifications 
The second file, predict.py, uses a trained network to predict the class for an input image. 

Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top K most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu

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
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from collections import OrderedDict


################################################################################################################
###### Global variables and hyperparameters
#in actual project, data is in 'flowers'
parser = argparse.ArgumentParser(description='predict.py: Opens a pretrained network and predicts an image class.')
parser.add_argument('--input', help='Path to input image file', required=True)
parser.add_argument('--checkpoint', help='Path to checkpoint', required=True)
parser.add_argument('--top_k', help='Display the top k most likely classes for the image', required=False)
parser.add_argument('--category_names', help='JSON file with the category label mapping', required=False)
parser.add_argument('--gpu', help='Choice of GPU or CPU', required=False)

args = vars(parser.parse_args())

input_path = args['input'] 
checkpoint_file = args['checkpoint']

if args['top_k']:
	top_k = int(args['top_k'])
else:
	top_k = 3
	
if args['category_names']:
	category_names = args['category_names']
else:
	category_names = 'cat_to_name.json'

if args['gpu']:
	device = 'cuda'
else:
	device = 'cpu'
	


################################################################################################################
###### Load and build the model

num_classes = 102

def load(file):
    checkpoint = torch.load(file)    
    
    if checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    model.class_to_idx = checkpoint['class_to_idx']

    #rebuild the classifier architecture
    hidden_units = checkpoint['hidden_units']
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

    model.load_state_dict(checkpoint['state_dict'])
    
    # change to cuda or CPU as specified
    model.to(device)

    return model

model2 = load(checkpoint_file)

################################################################################################################
###### Preprocess the image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    
    #resize the images where the shortest side is 256 pixels, keeping the aspect ratio. 

    #im.thumbnail(size)   
    x, y = im.size
    shortest = min(x,y)
    ratio = shortest / 256
    newx = int(x / ratio)
    newy = int(y / ratio)
    
    im = im.resize(size = (newx, newy))
    
    #crop out the center 224x224 portion of the image.
    xmargin = (newx - 224) / 2
    ymargin = (newy - 224) / 2
    im = im.crop(box = (xmargin, ymargin, (xmargin+224), (ymargin+224)))    
    
    # Process a PIL image for use in a PyTorch model
    np_image = np.array(im)
    
    #scale everything down by 1/256 to convert range [0,255] to [0,1]
    np_image = np_image / np.array([256, 256, 256])
    
    ''' Normalize means to [0.485, 0.456, 0.406]
        normalize standard deviations to [0.229, 0.224, 0.225]
        Subtract the means from each color channel, then divide by the standard deviation'''
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    ''' color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
    reorder dimensions using ndarray.transpose. 
    The color channel needs to be first and retain the order of the other two dimensions.'''
    np_image = np_image.transpose((2,0,1))
    
    return np_image
	
#img_to_classify = process_image(input_path)

################################################################################################################
###### Predict the class

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #gpu version
    if device == 'cuda':
        im = process_image(image_path)
        tens = torch.from_numpy(im).type(torch.cuda.FloatTensor)
        tens.unsqueeze_(0)
        probs = F.softmax(model.forward(tens))
        top = probs.topk(topk)
        probs = torch.Tensor.detach(top[0]).cpu().numpy()
        classes = torch.Tensor.detach(top[1]).cpu().numpy()
        return probs[0], classes[0]
    #cpu version
    else:
        im = process_image(image_path)
        tens = torch.from_numpy(im).type(torch.FloatTensor)
        tens.unsqueeze_(0)
        probs = F.softmax(model.forward(tens))
        top = probs.topk(topk)
        probs = torch.Tensor.detach(top[0]).cpu().numpy()
        classes = torch.Tensor.detach(top[1]).cpu().numpy()
        return probs[0], classes[0]

probs, classes = predict(input_path, model2, top_k)  
print(probs)
print(classes)

################################################################################################################
###### Map the labels

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

Species = []
for i in classes:
    s = str(i + 1)
    Species.append(cat_to_name[s])

df = pd.DataFrame({'Classes':classes, 'Prob':probs, 'Species':Species})

tk = df.sort_values(by=['Prob'], ascending=1)

print("Top " + str(top_k) + " most likely classes for image:")
#for i in tk.iterrows():
#	print(i['Species'] + " : " + i['Prob'])
print(str(tk))
