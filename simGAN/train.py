# This code is adapted from
# https://github.com/wayaai/SimGAN/blob/master/utils/mpii_gaze_dataset_organize.py
# wich provide a Kreas implementation and partially copied from
# https://github.com/AlexHex7/SimGAN_pytorch/blob/master/main.py
# wich provide a PyTorch implementation that has been adaptated for our use.

import argparse
import os

import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torch.autograd import Variable
from PIL import Image
import torch
from torch import nn, optim

from models import Refiner, Discriminator
from dataset import DomainDataset
from image_history_buffer import ImageHistoryBuffer


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/ED/doc2medieval', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--nb_features', type=int, default=64, help='number of refiner features')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--delta', type=float, default=0.1, help='l1 delta parameter')
parser.add_argument('--transform_setting', type=str, default='raw')
opt = parser.parse_args()
print(opt)

# Test GPU availability and set it
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


###### Definition of variables ######
### Networks
refiner = Refiner(opt.input_nc, opt.nb_features)
discriminator = Discriminator(opt.input_nc)


if opt.cuda:
    refiner.cuda()
    discriminator.cuda()

#### Losses

# L1 loss function for the refiner
# n.b: size_average allow the further multiplication by the delta for a
#      given batch size

criterion_regularization_loss = nn.L1Loss(size_average=False)

# Local adversarial loss
#TODO: Future work, try as parameter reduction = 'sum' instead of mean
criterion_adversarial_loss = nn.CrossEntropyLoss()

#### Optimizer
refiner_optim = optim.SGD(refiner.parameters(), lr=opt.lr)
discriminator_optim = optim.SGD(discriminator.parameters(), lr=opt.lr)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_syn = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_real = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)


# Dataset loader
if opt.transform_setting == 'raw':
    transforms_ = [ transforms.Resize(int(opt.size * 1.12)),
                    transforms.RandomCrop(opt.size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

elif opt.transform_setting == 'rc':
    transforms_ = [ transforms.Resize([3086, 2132]),
                    transforms.CenterCrop([1886, 1232]),
                    transforms.RandomCrop(opt.size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataloader = DataLoader(DomainDataset(opt.dataroot, transforms_=transforms_),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu,
                        pin_memory=True)

###############################################################################
########################### Pre-training functions ############################
###############################################################################

### Refiner pre-training
def pre_train_r(step_nbr):
    print('Pre-train the refiner network %d times...' % step_nbr)

    for index in range(step_nbr):
        for i, batch in enumerate(dataloader):
            syn_img_batch = Variable(input_syn.copy_(batch['S']))

            refiner.train()
            ref_img_batch = refiner(syn_img_batch)

            r_loss = criterion_regularization_loss(ref_img_batch, syn_img_batch)
            r_loss = torch.mul(r_loss, opt.delta)
            refiner_optim.zero_grad()
            r_loss.backward()
            refiner_optim.step()

            # log
            if (i  == len(dataloader) - 1):
                print('[%d/%d] Refiner loss: %.4f' % (index+1, step_nbr, r_loss.data[0]))

                syn_img_batch = Variable(input_syn.copy_(batch['S']))
                real_img_batch = Variable(input_real.copy_(batch['T']))
                refiner.eval()

                if not os.path.exists('results/pretrain_res/'):
                    os.makedirs('results/pretrain_res/')

                save_image(syn_img_batch, 'results/pretrain_res/syn%04d.png' % (index+1))
                save_image(real_img_batch, 'results/pretrain_res/real%04d.png' % (index+1))
                save_image(ref_img_batch, 'results/pretrain_res/ref%04d.png' % (index+1))

                refiner.train()

    print('Save the pretrained refiner to models/refiner_pre.pth')
    torch.save(refiner.state_dict(), 'models/refiner_pre.pth')



def pre_train_d(step_nbr):

    print('Pre-train the discriminator network %d times...' % step_nbr)

    discriminator.train()
    refiner.eval()

    for index in range(step_nbr):
        for i, batch in enumerate(dataloader):
            real_img_batch = Variable(input_real.copy_(batch['T']))
            syn_img_batch = Variable(input_syn.copy_(batch['S']))
            discriminator_optim.zero_grad()
            assert real_img_batch.size(0) == syn_img_batch.size(0)

            # Real image D
            d_real_pred = discriminator(real_img_batch).view(-1, 2)

            if opt.cuda:
                d_real_y = Variable(torch.zeros(d_real_pred.size(0)).type(torch.LongTensor)).cuda()
                d_ref_y = Variable(torch.ones(d_real_pred.size(0)).type(torch.LongTensor)).cuda()

            else:
                d_real_y = Variable(torch.zeros(d_real_pred.size(0)).type(torch.LongTensor))
                d_ref_y = Variable(torch.ones(d_real_pred.size(0)).type(torch.LongTensor))

            d_loss_real = criterion_adversarial_loss(d_real_pred, d_real_y)

            # syn image D
            ref_img_batch = refiner(syn_img_batch)
            d_ref_pred = discriminator(ref_img_batch).view(-1, 2)

            d_loss_ref = criterion_adversarial_loss(d_ref_pred, d_ref_y)
            d_loss = d_loss_real + d_loss_ref

            d_loss.backward()
            discriminator_optim.step()

            if (i == len(dataloader) - 1):
                print('[%d/%d] (D)d_loss:%f  '
                      % (index+1, step_nbr, d_loss.data[0]))

    print('Save Discriminator_pretrained model to models/discriminator_pre.pth')
    torch.save(discriminator.state_dict(), 'models/discriminator_pre.pth')


###############################################################################
############################# Training function ###############################
###############################################################################

def train(r_steps, d_steps):
    print('Training...')
    image_history_buffer = ImageHistoryBuffer((0, opt.input_nc, opt.size, opt.size),
                                              opt.batchSize * 1000, opt.batchSize )

    for epoch in range(opt.epoch, opt.n_epochs):
        print('Epoch[%d/%d]' % (epoch, opt.n_epochs))

        # Train refiner
        discriminator.eval()
        refiner.train()

        for p in discriminator.parameters():
            p.requires_grad = False


        s_total_r_loss = 0.0
        s_total_r_loss_reg_scale = 0.0
        s_total_r_loss_adv = 0.0

        # Train Refiner during 25 steps
        for index in range(r_steps):
            total_r_loss = 0.0
            total_r_loss_reg_scale = 0.0
            total_r_loss_adv = 0.0

            for i, batch in enumerate(dataloader):
                syn_img_batch = Variable(input_syn.copy_(batch['S']))
                refiner_optim.zero_grad()
                discriminator_optim.zero_grad()

                ref_img_batch = refiner(syn_img_batch)
                d_ref_pred = discriminator(ref_img_batch).view(-1, 2)

                if opt.cuda:
                    d_real_y = Variable(torch.zeros(d_ref_pred.size(0)).type(torch.LongTensor)).cuda()
                else:
                    d_real_y = Variable(torch.zeros(d_ref_pred.size(0)).type(torch.LongTensor))

                r_loss_reg_scale = criterion_regularization_loss(ref_img_batch, syn_img_batch)
                r_loss_reg_scale = torch.mul(r_loss_reg, opt.delta)
                r_loss_adv = criterion_adversarial_loss(d_ref_pred, d_real_y)
                r_loss = r_loss_reg_scale + r_loss_adv
                r_loss.backward()
                refiner_optim.step()

                total_r_loss += r_loss
                total_r_loss_reg_scale += r_loss_reg_scale
                total_r_loss_adv += r_loss_adv

            s_total_r_loss += total_r_loss / len(dataloader)
            s_total_r_loss_reg_scale += total_r_loss_reg_scale /len(dataloader)
            s_total_r_loss_adv += total_r_loss_adv / len(dataloader)


        mean_r_loss = s_total_r_loss / r_steps
        mean_r_loss_reg_scale = s_total_r_loss_reg_scale / r_steps
        mean_r_loss_adv = s_total_r_loss_adv / r_steps
        print('Refiner: r_loss:%.4f, r_loss_reg:%.4f, r_loss_adv:%.4f'
                  %(mean_r_loss.data[0], mean_r_loss_reg_scale.data[0], mean_r_loss_adv.data[0]))


        # Train D during 25 steps
        refiner.eval()
        discriminator.train()
        for p in discriminator.parameters():
            p.requires_grad = True

        for index in range(d_steps):
            for i, batch in enumerate(dataloader):

                real_img_batch = Variable(input_real.copy_(batch['T']))
                syn_img_batch = Variable(input_syn.copy_(batch['S']))

                discriminator.zero_grad()

                assert real_img_batch.size(0) == syn_img_batch.size(0)

                ref_img_batch = refiner(syn_img_batch)

                # use a history of refined Images
                half_batch_from_img_hist = image_history_buffer.get_from_image_history_buffer()
                image_history_buffer.add_to_image_history_buffer(ref_img_batch.cpu().data.numpy())

                if len(half_batch_from_img_hist):
                    torch_type = torch.from_numpy(half_batch_from_img_hist)
                    v_type = Variable(torch_type).cuda()
                    ref_img_batch[opt.batchSize // 2] = v_type

                d_real_pred = discriminator(real_img_batch).view(-1, 2)
                if opt.cuda:
                    d_real_y = Variable(torch.zeros(d_real_pred.size(0)).type(torch.LongTensor)).cuda()
                else:
                    d_real_y = Variable(torch.zeros(d_real_pred.size(0)).type(torch.LongTensor))
                d_loss_real = criterion_adversarial_loss(d_real_pred, d_real_y)

                d_ref_pred = discriminator(ref_img_batch).view(-1, 2)
                if opt.cuda:
                    d_ref_y = Variable(torch.ones(d_ref_pred.size(0)).type(torch.LongTensor)).cuda()
                else:
                    d_ref_y = Variable(torch.ones(d_ref_pred.size(0)).type(torch.LongTensor))

                d_loss_ref = criterion_adversarial_loss(d_ref_pred, d_ref_y)

                d_loss = d_loss_real + d_loss_ref
                d_loss.backward()
                discriminator_optim.step()

                if (i == len(dataloader)-1):
                    print('Discriminator: d_loss:%.4f, real_loss:%.4f, refine_loss:%.4f'
                        % (d_loss.data[0] / 2, d_loss_real.data[0], d_loss_ref.data[0]))

        # log every `log_interval` steps
        if (epoch % 2 == 0):
            print('Save models')
            real_img_batch =  Variable(input_real.copy_(dataloader.__iter__().next()['T']))
            syn_img_batch =  Variable(input_syn.copy_(dataloader.__iter__().next()['S']))

            refiner.eval()
            ref_img_batch = refiner(syn_img_batch)

            if not os.path.exists('results/train_res/'):
                os.makedirs('results/train_res/')

            save_image(syn_img_batch, 'results/train_res/syn%04d.png' % (epoch+1))
            save_image(real_img_batch, 'results/train_res/real%04d.png' % (epoch+1))
            save_image(ref_img_batch, 'results/train_res/ref%04d.png' % (epoch+1))


    torch.save(refiner.state_dict(), 'models/refiner.pth')
    torch.save(discriminator.state_dict(), 'models/discriminator.pth')

if __name__ == '__main__':
    pre_train_r(20)
    pre_train_d(5)
    train(25,1)
