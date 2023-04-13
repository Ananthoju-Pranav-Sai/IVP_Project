import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 4096)
        self.deconv1 = nn.ConvTranspose2d(256, 192, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(192, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 48, kernel_size=4, stride=2, padding=1)
        self.pixelconv = nn.Conv2d(48, opt.outViewN * 4, kernel_size=1, stride=1)

    def forward(self, latent):
        feat = F.relu(latent)
        feat = F.relu(self.fc1(feat))
        feat = F.relu(self.fc2(feat))
        feat = F.relu(self.fc3(feat))
        feat = feat.view(self.opt.batchSize, -1, 4, 4)
        feat = F.relu(self.deconv1(feat))
        feat = F.relu(self.deconv2(feat))
        feat = F.relu(self.deconv3(feat))
        feat = F.relu(self.deconv4(feat))
        feat = F.relu(self.deconv5(feat))
        feat = self.pixelconv(feat)
        XYZ, maskLogit = torch.split(feat, [self.opt.outViewN * 3, self.opt.outViewN], dim=1)
        return XYZ, maskLogit
