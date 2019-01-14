import torch as t
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random



### input file


class DataSet(Dataset):
    def __init__(self, file):
        self.file = file
        with open(self.file) as f:
            self.datas = f.readlines()

    def __len__(self):
        return

    def __getitem__(self, item):
        return





