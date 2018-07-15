#!/usr/bin/env python3
"""
@author: Stefan Filipiak
"""

import argparse
import json
from time import time, strftime, gmtime

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

def main():
    start_time = time()

    in_arg = get_input_args()
    
    device = torch.device("cuda:0" if in_arg.gpu and torch.cuda.is_available() else "cpu")
    print("Predicting on device:", device)
    
    checkpoint = load_checkpoint(in_arg.checkpoint)
    model = checkpoint['model']
    model.state_dict = checkpoint['state_dict']
    model.to(device)
    
    if device == 'cuda:0':
        model.cuda()
    else:
        model.cpu()
    
    probs, classes = predict(in_arg.input, model, device, in_arg.top_k)
    probs = probs.cpu().numpy()
    classes = classes.cpu().numpy()
    
    if in_arg.category_names == '':
        class_names = classes
    else:
        class_names = get_classes_as_names(classes, checkpoint["class_to_idx"], in_arg.category_names)
    
    data = pd.DataFrame({ 'Flower': class_names, 'Probability': probs })
    data = data.sort_values('Probability', ascending=False)
    
    print("Predicting image:", in_arg.input)
    print()
    print(data)
    
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
    
    parser.add_argument('input', type=str, help='Path to the image that should be predicted.')
    parser.add_argument('checkpoint', type=str, help='checkpoint to load')
    parser.add_argument('--top_k', type=int, default=3, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='', help='File witch includes the mapping of categories to rel names')
    parser.add_argument('--gpu', type=bool, default=False, const=True, nargs='?', help='train on gpu')

    return parser.parse_args()


def load_checkpoint(checkpoint):
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    
    return checkpoint

def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Tensor
    """
    
    im = Image.open(image)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return preprocess(im)


def predict(image_path, model, device, topk=3):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    
    # Implement the code to predict the class from an image file
    im_torch = process_image(image_path)
    
    im_torch.unsqueeze_(0)
    im_torch.requires_grad_(False)
    im_torch.to(device)
    
    if device == 'cuda:0':
        im_torch.cuda()
    else:
        im_torch.cpu()
    
    model.eval()
    
    with torch.no_grad():
        output = model(im_torch)
        results = torch.exp(output).topk(topk)
    
    probs = results[0][0]
    classes = results[1][0]
    
    return probs, classes


def get_classes_as_names(classes, class_to_idx, category_names):
    """
    Converts class numbers into category names
    """
    names = {}
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    for k in class_to_idx:
        names[class_to_idx[k]] = cat_to_name[k]
        
    return [names[c] for c in classes]


if __name__ == '__main__':
    main()
