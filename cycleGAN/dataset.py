import glob
import os
import random
from PIL import Image
import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DomainDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.inputs_S = sorted(glob.glob(os.path.join(root, '%s/S' % mode) + '/*.*'))
        self.inputs_T = sorted(glob.glob(os.path.join(root, '%s/T' % mode) + '/*.*'))

    def __getitem__(self, index):
        input_S = self.transform(Image.open(self.inputs_S[index % len(self.inputs_S)]))
        input_T = self.transform(Image.open(self.inputs_S[index % len(self.inputs_T)]))

        return {'S': input_S, 'T': input_T}

    def __len__(self):
        return max(len(self.inputs_S), len(self.inputs_T))
