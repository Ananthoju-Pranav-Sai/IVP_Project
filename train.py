from dataset import ObjectDataset
from encoder import Encoder
from decoder import Decoder
from transform import fuse3D
from utils import loss as model_loss
import lightning.pytorch as pl
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


class opt_class:
    input_channel = 0
    latentDim = 0
    outputViewN = 0
    batchSize = 0
    Khom2Dto3D = 0
    renderDepth = 0
    outH = 0
    outW = 0
    novelN = 0
    upscale = 0
    Lambda = 0


opt = opt_class()
fusetrans = 0
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
        x, y = batch
        point_cloud = self.model(x)
        loss = model_loss(opt, point_cloud, y, renderTrans)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    pl_model = reconstruction_model()

    train_dataset = ObjectDataset(chunk_size=50, train=True)
    test_dataset = ObjectDataset(chunk_size=50, train=False)
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset)

    trainer = pl.Trainer(max_epochs=100000, accelerator="gpu")
    trainer.fit(model=pl_model, train_dataloaders=train_loader)

    trainer.test(model=pl_model, dataloaders=test_loader)


if __name__ == '__main__':
    main()
