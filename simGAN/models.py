#  This code is adapted from the Kreas code
#  https://github.com/wayaai/SimGAN/blob/master/utils/mpii_gaze_dataset_organize.py
#

"""
PyTorch Implementation of the networks mainly following the `3.1 Appearance-based Gaze Estimation` from
[Learning from Simulated and Unsupervised Images through Adversarial Training]
(https://arxiv.org/pdf/1612.07828v1.pdf), networks recomendations
Note: Only Python 3 support currently.
"""


import torch.nn as nn
import torch.nn.functional as F

# We add the relfection padding at the top of the resnet block becouse it shown b
# better results in several other unsupervised iamge-to-image GANs implementations
# We also ad instance normalization for the same reasons.
class ResiduaBlock(nn.Module):
    def __init__(self, in_features):
        super(ResiduaBlock, self).__init__()

        conv_block =  [ nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features) ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

# We change the number of res_block from 4 to 6 regarding the complexity of our dataset
class Refiner(nn.Module):
    def __init__(self, input_nc, nb_features, n_residual_blocks=9):
        super(Refiner, self).__init__()


        # Initial convolutional block
        model = [ nn.ReflectionPad2d(3),
                 # Originally kernel size of 7, we change the kernel size to allow the network
                 # to skip the details in the images in order to perform a more agressive refining
                  nn.Conv2d(input_nc, 64, 13),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(inplace=True) ]


        # Residual blocks number of 9 same as in cycleGAN, to make our loss assumption stronger.
        for _ in range(n_residual_blocks):
            model += [ResiduaBlock(nb_features)]

        # Output layer
        # output_nc = img_channel = 3
        # return refined img 1 x 1 (kernel size)
        model += [ nn.ReflectionPad2d(3),
                   nn.Conv2d(64, input_nc, 1, 1),
                   nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Same as in original implementation
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another

        model = [ nn.Conv2d(input_nc, 96, 7, stride=4, padding=1),
                  nn.ReLU(inplace=True) ]

        model += [ nn.Conv2d(96, 64, 5, stride=2, padding=1),
                   nn.ReLU(inplace=True),
                   nn.MaxPool2d(3, stride=2, padding=1)]

        model += [ nn.Conv2d(64, 32, 3, stride=2, padding=1),
                   nn.ReLU(inplace=True) ]

        model += [ nn.Conv2d(32, 32, 1, stride=1, padding=1),
                   nn.ReLU(inplace=True) ]

        model += [ nn.Conv2d(32, 2, 1, stride=1, padding=1),
                   nn.ReLU(inplace=True) ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        output = x.view(x.size(0), -1, 2)
        return output
