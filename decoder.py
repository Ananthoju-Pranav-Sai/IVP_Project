import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Decoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.fc1 = self.linearLayer(512, 1024)
        self.fc2 = self.linearLayer(1024, 2048)
        self.fc3 = self.linearLayer(2048, 4096)
        self.deconv1 = self.deconv2Layer(256, 192)
        self.deconv2 = self.deconv2Layer(192, 128)
        self.deconv3 = self.deconv2Layer(128, 96)
        self.deconv4 = self.deconv2Layer(96, 64)
        self.deconv5 = self.deconv2Layer(64, 48)
        self.pixelconv = nn.Conv2d(48, opt.outViewN * 4, kernel_size=1, stride=1)

        X, Y = np.meshgrid(range(self.opt.outW), range(self.opt.outH), indexing="xy")  # [H,W]
        X, Y = X.astype(np.float32), Y.astype(np.float32)
        initTile = np.concatenate([np.tile(X, [self.opt.outViewN, 1, 1]),
                                   np.tile(Y, [self.opt.outViewN, 1, 1]),
                                   np.ones([self.opt.outViewN, self.opt.outH, self.opt.outW],
                                           dtype=np.float32) * self.opt.renderDepth,
                                   np.zeros([self.opt.outViewN, self.opt.outH, self.opt.outW], dtype=np.float32)],
                                  axis=0)  # [4V,H,W]
        biasInit = np.expand_dims(np.transpose(initTile, axes=[1, 2, 0]), axis=0)  # [1,H,W,4V]
        biasInit = torch.tensor(biasInit).permute([0, 3, 1, 2])
        self.biasInit = nn.Parameter(biasInit)

    def deconv2Layer(self, in_channels, out_channels):
        conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                  padding=1, output_padding=1)
        batchnorm = nn.BatchNorm2d(num_features=out_channels)
        return nn.Sequential(conv, batchnorm)

    def linearLayer(self, in_features, out_features):
        fc = nn.Linear(in_features=in_features, out_features=out_features)
        batchnorm = nn.BatchNorm1d(num_features=out_features)
        return nn.Sequential(fc, batchnorm)

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
        feat = self.pixelconv(feat)+self.biasInit
        XYZ, maskLogit = torch.split(feat, [self.opt.outViewN * 3, self.opt.outViewN], dim=1)
        return XYZ, maskLogit
