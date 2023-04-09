import torch

from dataset import ObjectDataset
from encoder import Encoder
from decoder import Decoder
from transform import fuse3D
from utils import loss as model_loss
import lightning.pytorch as pl
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np


class opt_class:
    input_channel = 3
    outViewN = 8
    batchSize = 20
    Khom2Dto3D = np.array([[64, 0, 0, 64 / 2],
                           [0, -64, 0, 64 / 2],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
    renderDepth = 1.0
    outH = 128
    outW = 128
    novelN = 5
    upscale = 5
    Lambda = 1.0
    H = 128
    W = 128


opt = opt_class()
fusetrans = np.load(f"data/trans_fuse{opt.outViewN}.npy")
fusetrans /= np.linalg.norm(axis=1)[:, np.newaxis]
renderTrans = 0


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

    def forward(self, x):
        x = self.encoder(x)
        xyz, maskLogit = self.decoder(x)
        x = fuse3D(opt, xyz, maskLogit, fusetrans)
        return x


class reconstruction_model(pl.LightningModule):
    def __init__(self):
        super(reconstruction_model, self).__init__()
        self.model = model()

    def training_step(self, batch, batch_idx):
        images, depths, trans, masks = batch
        point_cloud = self.model(images)
        loss = model_loss(opt, point_cloud, (depths, masks), trans)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    pl_model = reconstruction_model()

    train_dataset = ObjectDataset(chunk_size=50, train=True)
    test_dataset = ObjectDataset(chunk_size=50, train=False)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize)

    trainer = pl.Trainer(max_epochs=10000, accelerator="gpu")
    trainer.fit(model=pl_model, train_dataloaders=train_loader)

    trainer.test(model=pl_model, dataloaders=test_loader)


if __name__ == '__main__':
    main()
