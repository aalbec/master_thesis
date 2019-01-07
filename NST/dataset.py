import glob
import os
import random
from PIL import Image

import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(root + '/*'))


    def __getitem__(self, index):

        file_name = self.files[index]
        item = self.transform(Image.open(self.files[index]))


        # Set labels for classification
        if 'cy' in file_name:
          label = torch.from_numpy(np.array([0]))
        elif 'girart' in file_name:
          label = torch.from_numpy(np.array([1]))
        elif 'justinien' in file_name:
          label = torch.from_numpy(np.array([2]))
        else:
          print('Error, incorrect file name, index' + str(index))
          sys.exit(0)

        return {'img': item, 'label': label}

    def __len__(self):
        return len(self.files)
