import torch.nn as nn
import torch.nn.functional as F
import utils

class residual_block(nn.Module):
    def __init__(self, channel):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(channel)
        utils.initialize_weights(self)

    def forward(self, inputs):
        residual = F.relu(self.norm1(self.conv1(inputs)))
        residual = self.norm2(self.conv2(residual))
        return inputs + residual


class generator(nn.Module):
    def __init__(self, noise_dim=100, out_channel=1, filters=64, res_num=4):
        super(generator, self).__init__()
        
        self.fc = nn.Linear(noise_dim, filters * 8 * 8)
        
        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(filters, filters, kernel_size=4, stride=2, padding=1), 
            nn.InstanceNorm2d(filters),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(filters, filters // 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(filters // 2),
            nn.ReLU(inplace=True),
        )
        
        self.res_blocks = nn.Sequential(*[residual_block(filters // 2) for _ in range(res_num)])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(filters // 2, out_channel, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )
        utils.initialize_weights(self)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 8, 8)
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        return x


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
        utils.initialize_weights(self)

    def forward(self, img):
        validity = self.model(img)
        validity = validity.squeeze(1).squeeze(2)
        return validity