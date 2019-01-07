import os
import glob
import numpy as np
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import ImageDataset


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='NST/models/')
parser.add_argument('--dataroot', type=str, default='datasets/TDD', help='root directory of the dataset')
parser.add_argument('--img_setting', type=str, default='raw')
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


if opt.img_setting == 'raw':
    img_transform = [
        transforms.Resize(int(opt.img_size*1.12)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

train_dataset = ImageDataset(opt.dataroot, transforms_=img_transform)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size,
                              shuffle=True, num_workers=8)


# Import VGG19 Model
vgg19 = torchvision.models.vgg19(pretrained=False)

# Number of filters in the bottleneck layer
num_ftrs = vgg19.classifier[6].in_features

# convert all the layers to list and remove the last one
features = list(vgg19.classifier.children())[:-1]

## Add the last layer based on the num of classes in our dataset
features.extend([nn.Linear(num_ftrs, opt.num_classes)])

## convert it into container and add it to our model class.
vgg19.classifier = nn.Sequential(*features)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vgg19.parameters(), momentum=opt.momentum,
                     lr = opt.lr)

if opt.cuda:
   vgg19.cuda()


for epoch in range(opt.num_epochs):
  running_loss = 0.0

  for i, data in enumerate(train_dataloader, 0):

    # get the inputs
    inputs, labels = data['img'], data['label']
    if opt.cuda:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda().squeeze(1))
    else:
        inputs = Variable(inputs)
        labels = Variable(labels.squeeze(1))
    # zero the parameter gradient
    optimizer.zero_grad()


    #================Forward Pass===============#

    outputs = vgg19(inputs)
    loss = criterion(outputs, labels)

    #===============Backward Pass===============#

    loss.backward()
    optimizer.step()

  #====================Log======================#
    running_loss += loss.data[0]
    if i % 150 == 149:    # print the loss every 100 mini-batches
      print('Epoch [{}/{}]: loss:{:.4f}'
            .format(epoch + 1, opt.num_epochs, running_loss / 150))

if opt.img_setting == 'raw':
    torch.save(vgg19.state_dict(), opt.model_path + 'vgg19_pretrained_rc.pth')
elif opt.img_setting =='rc':
    torch.save(vgg19.state_dict(), opt.model_path + 'vgg19_pretrained_raw.pth')
