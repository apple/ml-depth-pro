import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
   