from data import trainloader, testloader
from layers import encoder, decoder
from transforms import generate_point_cloud
from utils import model_loss
import lightning.pytorch as pl
import torch.nn as nn

def model(nn.module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()
        self.generate_point_cloud = generate_point_cloud()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.generate_point_cloud(x)
        return x

def reconstruction_model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model()

    def training_step(self, batch, batch_idx):
        x, y = batch
        point_cloud = self.model(x)
        loss = model_loss(point_cloud, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

pl_model = reconstruction_model()

train_loader = trainloader()
test_loader = testloader()

trainer = pl.Trainer(max_epochs=100000)
trainer.fit(model=pl_model, train_dataloaders=train_loader, accelerator="gpu")

trainer.test(test_dataloaders=test_loader)
