from train import reconstruction_model, opt_class
from dataset import ObjectDataset
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import numpy as np


opt = opt_class()


def main():
    pl_model = reconstruction_model.load_from_checkpoint(
        "lightning_logs/version_13/checkpoints/epoch=1000-step=120120.ckpt")

    test_dataset = ObjectDataset(chunk_size=50, train=False)
    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, num_workers=4)

    trainer = pl.Trainer()

    outs = trainer.predict(model=pl_model, dataloaders=test_loader)
    for i, (a, b, c) in enumerate(outs):
        for j in range(len(a)):
            np.save(f"outputs/image_{i*opt.batchSize+j+1}", a[j].numpy())
            np.save(f"outputs/xyz_{i*opt.batchSize+j+1}", b[j].numpy())
            np.save(f"outputs/ml_{i*opt.batchSize+j+1}", c[j].numpy())


if __name__ == '__main__':
    main()
