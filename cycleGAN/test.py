# Code is apdapted from https://github.com/aitorzip/PyTorch-CycleGAN
# which is  basically a cleaner and less obscured implementation of pytorch-CycleGAN-and-pix2pix.
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# All credit goes to the authors of CycleGAN, Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A.
# https://arxiv.org/pdf/1703.10593.pdf
import argparse
import argparse
import sys
import os
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from dataset import DomainDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_S2T_raw', type=str, default='cycleGAN/models/netG_S2T_raw.pth', help='S2T generator checkpoint file')
parser.add_argument('--generator_T2S_raw', type=str, default='cycleGAN/models/netG_T2S_raw.pth', help='T2S generator checkpoint file')
parser.add_argument('--generator_S2T_rc', type=str, default='cycleGAN/models/netG_S2T_rc.pth', help='S2T generator checkpoint file')
parser.add_argument('--generator_T2S_rc', type=str, default='cycleGAN/models/netG_T2S_rc.pth', help='T2S generator checkpoint file')
parser.add_argument('--training_mode', type=str, default='scratch')
parser.add_argument('--image_setting', type=str, default='raw')

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_S2T = Generator(opt.input_nc, opt.output_nc)
netG_T2S = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_S2T.cuda()
    netG_T2S.cuda()

# Load state dicts
if opt.image_setting =='raw':
    netG_S2T.load_state_dict(torch.load(opt.generator_S2T_raw))
    netG_T2S.load_state_dict(torch.load(opt.generator_T2S_raw))

elif opt.image_setting == 'rc':
    netG_S2T.load_state_dict(torch.load(opt.generator_S2T_rc))
    netG_T2S.load_state_dict(torch.load(opt.generator_T2S_rc))

# Set model's test mode
netG_S2T.eval()
netG_T2S.eval()

# Inputs
Tensor = torch.cuda.FloatTensor
input_S = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_T = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
if opt.image_setting == 'raw':
    transforms_ = [ transforms.Resize(int(opt.size * 1.12)),
                    transforms.RandomCrop(opt.size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

elif opt.image_setting == 'rc':
    transforms_ = [ transforms.Resize([3086, 2132]),
                    transforms.CenterCrop([1886, 1232]),
                    transforms.RandomCrop(opt.size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
###################################

###### Testing######

# Create output dirs if they don't exist
if opt.image_setting == 'raw':
        if not os.path.exists('results/S/raw'):
            os.makedirs('results/S/raw')
        if not os.path.exists('results/T/raw'):
            os.makedirs('results/T/raw')

elif opt.image_setting == 'rc':
    if opt.training_mode == 'scratch':
        if not os.path.exists('results/S/rc/scratch'):
            os.makedirs('results/S/rc/scratch')
        if not os.path.exists('results/T/rc/scratch'):
            os.makedirs('results/T/rc/scratch')

    elif opt.training_mode == 'pretrained':
        if not os.path.exists('results/S/rc/pretrained'):
            os.makedirs('results/S/raw/pretrained')
        if not os.path.exists('results/T/rc/pretrained'):
            os.makedirs('results/T/raw/pretrained')



for i, batch in enumerate(dataloader):
    # Set model input
    real_S = Variable(input_S.copy_(batch['S']))
    real_T = Variable(input_T.copy_(batch['T']))

    # Generate output
    fake_T = 0.5*(netG_S2T(real_S).data + 1.0)
    fake_S = 0.5*(netG_T2S(real_T).data + 1.0)

    # Save image files
    if opt.image_setting == 'raw':
        save_image(fake_S, 'results/S/raw/%04d.png' % (i+1))
        save_image(fake_T, 'results/T/raw/rfaked_%04d.png' % (i+1))
        save_image(real_T, 'results/T/raw/real_target_%04d.png' % (i+1))
        save_image(real_S, 'results/T/raw/real_source_%04d.png' % (i+1))

    elif opt.image_setting == 'rc':
        if opt.training_mode == 'scratch':
            save_image(fake_S, 'results/S/rc/scratch/%04d.png' % (i+1))
            save_image(fake_T, 'results/S/rc/scratch/rfaked_%04d.png' % (i+1))
            save_image(real_T, 'results/S/rc/scratch/real_target_%04d.png' % (i+1))
            save_image(real_S, 'results/S/rc/scratch/real_source_%04d.png' % (i+1))
        elif opt.training_mode == 'pretrained':
            save_image(fake_S, 'results/S/rc/pretrained/%04d.png' % (i+1))
            save_image(fake_T, 'results/S/rc/pretrained/rfaked_%04d.png' % (i+1))
            save_image(real_T, 'results/S/rc/pretrained/real_target_%04d.png' % (i+1))
            save_image(real_S, 'results/S/rc/pretrained/real_source_%04d.png' % (i+1))


    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
