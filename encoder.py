import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.conv1 = self.conv2Layer(int(opt.input_channel), 96)
        self.conv2 = self.conv2Layer(96, 128)
        self.conv3 = self.conv2Layer(128, 192)
        self.conv4 = self.conv2Layer(192, 256)
        self.fc1 = self.linearLayer(256 * 4 * 4, 2048)
        self.fc2 = self.linearLayer(2048, 1024)
        self.fc3 = self.finalLayer(1024, 512)

    def conv2Layer(self, in_channels, out_channels):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        batchnorm = nn.BatchNorm2d(num_features=out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, batchnorm, relu)

    def linearLayer(self, in_features, out_features):
        fc = nn.Linear(in_features=in_features, out_features=out_features)
        batchnorm = nn.BatchNorm1d(num_features=out_features)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(fc, batchnorm, relu)

    def finalLayer(self, in_features, out_features):
        fc = nn.Linear(in_features=in_features, out_features=out_features)
        return nn.Sequential(fc)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [B,H,W,3] -> [B,3,H,W]
        feat = x
        feat = self.conv1(feat)  # 32x32
        feat = self.conv2(feat)  # 16x16
        feat = self.conv3(feat)  # 8x8
        feat = self.conv4(feat)  # 4x4
        feat = feat.reshape([-1, 256 * 4 * 4])
        feat = self.fc1(feat)
        feat = self.fc2(feat)
        feat = self.fc3(feat)
        latent = feat
        return latent
