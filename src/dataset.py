import numpy as np 
import torch 
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

class EthTransGraphDataset(Dataset):
  """Pytorch_geometric Dataset for graph data"""
  def __init__(self, data, window_size=10, target_window_size=5):
    """
    Args:
      data: pd.DataFrame, generated from gen_data function
      window_size: int, length of input data series
      target_window_size: int, length of predict interval
    """
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
    """Gets a pair of input and label
    
    Args:
      idx: int, index of pair in all data.
    
    Returns:
      data_window, target: (Data, float), Data instance of Pytorch_geometric 
                           and float denotes minimum gas price
    """
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