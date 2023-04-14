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
    batchSize = 10
    Khom2Dto3D = np.array([[64, 0, 0, 64 / 2],
                           [0, -64, 0, 64 / 2],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
    Khom3Dto2D = np.array(np.array([[128, 0, 0, 128 / 2],
                                    [0, -128, 0, 128 / 2],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32))
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
fusetrans /= np.linalg.norm(fusetrans, axis=1)[:, np.newaxis]


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

    def forward(self, x, device):
        x = self.encoder(x)
        xyz, maskLogit = self.decoder(x)
        x = fuse3D(opt, xyz, maskLogit, fusetrans, device)
        return x


class reconstruction_model(pl.LightningModule):
    def __init__(self):
        super(reconstruction_model, self).__init__()
        self.model = model()

    def training_step(self, batch, batch_idx):
        images, depths, trans, masks = batch
        point_cloud = self.model(images, self.device)
        loss1, loss2 = model_loss(opt, point_cloud, (depths, masks), trans, self.device)
        self.log("l1_loss", loss1, prog_bar=True, on_epoch=True)
        self.log("bce_loss", loss2, prog_bar=True, on_epoch=True)
        loss = loss1 + opt.Lambda * loss2
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        images, depths, trans, masks = batch
        a, b = self.model(images)
        return images, a, b

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)


def main():
    pl_model = reconstruction_model()

    train_dataset = ObjectDataset(chunk_size=50, train=True)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, num_workers=4, shuffle=True)

    trainer = pl.Trainer(max_epochs=10, accelerator="gpu")
    trainer.fit(model=pl_model, train_dataloaders=train_loader)

    test_dataset = ObjectDataset(chunk_size=50, train=False)
    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, num_workers=4)

    trainer.test(model=pl_model, dataloaders=test_loader)


if __name__ == '__main__':
    pl.seed_everything(422, workers=True)
    main()
