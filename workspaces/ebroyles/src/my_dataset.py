import os
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from constants import SAVE_ROOT

"""
from my_dataset import MyDataset as DS
train_filepaths = ["standard1/train.csv", ...]
test_filepath = ["standard1/test.csv", ...]
train_loader = DS.get_loader(train_filepaths, shuffle=True, batch_size=64, num_workers=63)
test_loader = DS.get_loader(test_filepaths, batch_size=64, num_workers=63)
"""

class MyDataset(Dataset):
    def __init__(self, filepath, relabel):
        self.filepath = os.path.join(SAVE_ROOT, filepath)
        self.data = np.loadtxt(self.filepath, delimiter=",", skiprows=1)
        self.num_features = self.data.shape[1] - 1
        self.relabel = relabel
        #apply the relabel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        X = row[:-1].astype(np.float32)
        y = int(row[-1])
        if y in self.relabel: y = self.relabel[y]
        return X, y

    @staticmethod
    def concat_ds(filepaths, relabel):
        return ConcatDataset([MyDataset(fp, relabel) for fp in filepaths])
        
    @staticmethod
    def get_loader(filepaths = [], relabel = {}, shuffle=False, batch_size=64, num_workers=63):
        #relabel = {0: 1, 4: 5} -> used to change the key label to the new label (Purpose: change labels to all 0 or 1)
        return DataLoader(MyDataset.concat_ds(filepaths, relabel), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)



        
        
