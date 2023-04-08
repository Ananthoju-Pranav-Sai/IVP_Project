import numpy as np
import torch
import scipy.io
from torch.utils.data import Dataset


class ObjectDataset(Dataset):
    def __init__(self, category="03001627", chunk_size=50, novelN =5, train=True) -> None:
        self.in_size = [64, 64]
        self.out_size = [128, 128]
        self.pred_size = [128, 128]
        self.chunk_size = chunk_size
        self.sampleN = 100
        self.input_viewN = 24
        self.out_viewN = 8
        self.novelN = novelN
        self.render_depth = 1.0

        if train:
            data_list = f"data/{category}_train.list"
        else:
            data_list = f"data/{category}_test.list"

        self.CADs = []
        with open(data_list) as file:
            for line in file:
                id = line.strip().split('/')[1]
                self.CADs.append(id)
        self.CADs.sort()

        self.data = {}
        idx = np.random.permutation(len(self.CADs))[:chunk_size]
        self.data["img"] = []
        self.data["depth"] = []
        self.data["trans"] = []
        self.data["mask"] = []

        for c in range(chunk_size):
            CAD = self.CADs[idx[c]]
            images = np.load(f"data/{category}_inputRGB/{CAD}.npy") / 255.0
            raw_data = scipy.io.loadmat(f"data/{category}_depth/{CAD}.mat")
            depth = raw_data["Z"]
            mask = depth != 0
            depth[~mask] = 1.0
            trans = raw_data["trans"]
            self.data["depth"].append(depth)
            self.data["trans"].append(trans)
            self.data["mask"].append(mask)
            for i in range(len(images)):
                self.data["img"].append(images[i])

        self.data["img"] = torch.tensor(np.array(self.data["img"], dtype=np.float32))
        self.data["depth"] = torch.tensor(np.array(self.data["depth"], dtype=np.float32))
        self.data["trans"] = torch.tensor(np.array(self.data["trans"], dtype=np.float32))
        self.data["mask"] = torch.tensor(np.array(self.data["mask"], dtype=bool))

    def __len__(self):
        return len(self.data["img"])

    def __getitem__(self, index):
        sample_idx = np.random.randint(self.sampleN,size=self.novelN)
        return [self.data["img"][index], self.data["depth"][index//self.input_viewN,sample_idx], \
                self.data["trans"][index//self.input_viewN,sample_idx], self.data["mask"][index//self.input_viewN,sample_idx]]
