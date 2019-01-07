# Code is apdapted from https://github.com/aitorzip/PyTorch-CycleGAN
# which is  basically a cleaner and less obscured implementation of pytorch-CycleGAN-and-pix2pix.
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# All credit goes to the authors of CycleGAN, Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A.
# https://arxiv.org/pdf/1703.10593.pdf
import argparse
import itertools

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image


from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
from dataset import DomainDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/ED/doc2medieval', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=25, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--model_path', type=str, default='cycleGAN/models/')
parser.add_argument('--training_mode', type=str, default='scratch')
parser.add_argument('--enc_S2T', type=str, default='cycleGAN/models/encoder_S2T.pth')
parser.add_argument('--enc_T2S', type=str, default='cycleGAN/models/encoder_T2S.pth')
parser.add_argument('--image_setting', type=str, default='raw')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Networks
netG_S2T = Generator(opt.input_nc, opt.output_nc)
netG_T2S = Generator(opt.output_nc, opt.input_nc)
netD_S = Discriminator(opt.input_nc)
netD_T = Discriminator(opt.output_nc)


if opt.cuda:
    netG_S2T.cuda()
    netG_T2S.cuda()
    netD_S.cuda()
    netD_T.cuda()

if opt.training_mode == 'scratch':
    # Init weights
    netG_S2T.apply(weights_init_normal)
    netG_T2S.apply(weights_init_normal)
    netD_S.apply(weights_init_normal)
    netD_T.apply(weights_init_normal)

elif opt.training_mode == 'pretrained':
    # Load state dicts
    netG_S2T.encoder.load_state_dict(torch.load(opt.enc_S2T))
    netG_T2S.encoder.load_state_dict(torch.load(opt.enc_T2S))


# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_S2T.parameters(), netG_T2S.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_S = torch.optim.Adam(netD_S.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_T = torch.optim.Adam(netD_T.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_S = torch.optim.lr_scheduler.LambdaLR(optimizer_D_S, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_T = torch.optim.lr_scheduler.LambdaLR(optimizer_D_T, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
if opt.cuda:
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

input_S = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_T = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_S_buffer = ReplayBuffer()
fake_T_buffer = ReplayBuffer()


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


dataloader = DataLoader(DomainDataset(opt.dataroot, transforms_=transforms_),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)


###### Training ######
for opt.epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_S = Variable(input_S.copy_(batch['S']))
        real_T = Variable(input_T.copy_(batch['T']))

        ###### Generators S2T and T2S ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_S2T(T) should equal T if real T is fed
        same_T = netG_S2T(real_T)
        loss_identity_T = criterion_identity(same_T, real_T)*5.0
        # G_T2S(S) should equal S if real S is fed
        same_S = netG_T2S(real_S)
        loss_identity_S = criterion_identity(same_S, real_S)*5.0

        # GAN loss
        fake_T = netG_S2T(real_S)
        pred_fake = netD_T(fake_T)
        loss_GAN_S2T = criterion_GAN(pred_fake, target_real.view(1,1))

        fake_S = netG_T2S(real_T)
        pred_fake = netD_S(fake_S)
        loss_GAN_T2S = criterion_GAN(pred_fake, target_real.view(1,1))

        # Cycle loss
        recovered_S = netG_T2S(fake_T)
        loss_cycle_STS = criterion_cycle(recovered_S, real_S)*10.0

        recovered_T = netG_S2T(fake_S)
        loss_cycle_TST = criterion_cycle(recovered_T, real_T)*10.0

        # Total loss
        loss_G = loss_identity_S + loss_identity_T + loss_GAN_S2T + loss_GAN_T2S + loss_cycle_STS + loss_cycle_TST
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator S ######
        optimizer_D_S.zero_grad()

        # Real loss
        pred_real = netD_S(real_S)
        loss_D_real = criterion_GAN(pred_real, target_real.view(1,1))

        # Fake loss
        fake_S = fake_S_buffer.push_and_pop(fake_S)
        pred_fake = netD_S(fake_S.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake.view(1,1))

        # Total loss
        loss_D_S = (loss_D_real + loss_D_fake)*0.5
        loss_D_S.backward()

        optimizer_D_S.step()
        ###################################

        ###### Discriminator T ######
        optimizer_D_T.zero_grad()

        # Real loss
        pred_real = netD_T(real_T)
        loss_D_real = criterion_GAN(pred_real, target_real.view(1,1))

        # Fake loss
        fake_T = fake_T_buffer.push_and_pop(fake_T)
        pred_fake = netD_T(fake_T.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake.view(1,1))

        # Total loss
        loss_D_T = (loss_D_real + loss_D_fake)*0.5
        loss_D_T.backward()

        optimizer_D_T.step()
        ###################################

        print('Epoch [{}/{}]: loss_G:{:.4f}'
              .format(epoch + 1, n_epochs, loss_G.data[0]))

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_S.step()
    lr_scheduler_D_T.step()

    # Save models checkpoints
    if opt.image_setting == 'raw':
        torch.save(netG_S2T.state_dict(), opt.model_path + 'netG_S2T_raw.pth')
        torch.save(netG_T2S.state_dict(), opt.model_path + 'netG_T2S_raw.pth')
        torch.save(netD_S.state_dict(), opt.model_path + 'netD_S_raw.pth')
        torch.save(netD_T.state_dict(), opt.model_path + 'netD_T_raw.pth')

    elif opt.image_setting == 'rc':
        torch.save(netG_S2T.state_dict(), opt.model_path + 'netG_S2T_rc.pth')
        torch.save(netG_T2S.state_dict(), opt.model_path + 'netG_T2S_rc.pth')
        torch.save(netD_S.state_dict(), opt.model_path + 'netD_S_rc.pth')
        torch.save(netD_T.state_dict(), opt.model_path + 'netD_T_rc.pth')
