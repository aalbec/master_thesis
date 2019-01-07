import argparse
import itertools
import glob

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
from PIL import Image


from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal

# Image Dataset building
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(root + '/*'))

    def __getitem__(self, index):
        item = self.transform(Image.open(self.files[index]))
        return item

    def __len__(self):
        return len(self.files)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/SDD', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--model_path', type=str, default='cycleGAN/models/')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

img_transform = [ transforms.Resize([3086, 2132]),
                transforms.CenterCrop([1886, 1232]),
                transforms.RandomCrop(opt.size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

train_dataset = ImageDataset(opt.dataroot, transforms_=img_transform)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize,
                              shuffle=True, num_workers=8)

model = Generator(opt.input_nc, opt.output_nc)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=opt.lr)

if opt.cuda:
   model.cuda()

for opt.epoch in range(opt.n_epochs):
    for data in train_dataloader:
        if opt.cuda:
            inputs = Variable(data).cuda()
        else:
            inputs = Variable(data)
            optimizer.zero_grad()


    #================Forward Pass===============#

        output = model(inputs)
        loss = criterion(output, inputs)

    #===============Backward Pass===============#

        loss.backward()
        optimizer.step()

    #====================Log======================#

    print('Epoch [{}/{}]: loss:{:.4f}'.format(epoch + 1, opt.n_epochs, loss.data[0]))
    input_img = inputs.data
    output_img = output.data

    if opt.dataroot == 'datasets/SDD':
        save_image(input_img, 'cycleGAN/results/ae/SDD' + '/input_img_epoch_' + str(epoch) + '.png')
        save_image(output_img, 'cycleGAN/results/ae/SDD' + '/output_img_epoch_' + str(epoch) + '.png')
    elif opt.dataroot == 'dataset/TDD':
        save_image(input_img, 'cycleGAN/results/ae/SDD' + '/input_img_epoch_' + str(epoch) + '.png')
        save_image(output_img, 'cycleGAN/results/ae/SDD' + '/output_img_epoch_' + str(epoch) + '.png')

if opt.dataroot == 'datasets/SDD':
    torch.save(model.encoder.state_dict(), 'cycleGAN/models/' + 'encoder_S2T.pth')
elif opt.dataroot == 'dataset/TDD':
    torch.save(model.encoder.state_dict(), 'cycleGAN/models/' + 'encoder_T2S.pth')
