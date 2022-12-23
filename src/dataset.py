import numpy as np 
import torch 
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

class EthTransGraphDataset(Dataset):
  def __init__(self, data, window_size=10, target_window_size=5):
    super().__init__()
    self.data = data
    self.tar = (self.data
                .apply(lambda row: row['data'].y, axis=1)
                .rolling(target_window_size)
                .min()
                .shift(-target_window_size)
                )
    self.window_size = window_size 
    self.target_window_size = target_window_size
    
  def len(self):
    return len(self.tar) - self.target_window_size
  def get(self, idx):
    low = idx + 1 - self.window_size
    if low < 0:
      idx = np.random.randint(self.window_size - 1, self.len())
    data_window = (self.data
                   .iloc[idx + 1 - self.window_size: idx + 1]
                   .values
                   .flatten())
    data_window = next(iter(
        DataLoader(data_window, batch_size=len(data_window), shuffle=False)
        ))
    target = self.tar.iloc[idx]
    data_window.inner_batch = torch.zeros_like(data_window.batch)
    return data_window, target