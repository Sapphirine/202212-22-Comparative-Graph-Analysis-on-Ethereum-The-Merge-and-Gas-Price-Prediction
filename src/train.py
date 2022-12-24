import os 
import torch 
from sklearn.model_selection import train_test_split
from torch_geometric.data.lightning.datamodule import LightningDataset
import pytorch_lightning as pl

from .config import data_file_name
from .dataset import EthTransGraphDataset
from .model import ETHGT

data_path = os.path.join(os.getcwd(), data_file_name)

def init_model_dataset(config, data_dir=None):
  """Initializes ETHGT model from searched best parameters"""
  window_size = config['window_size']
  batch_size = config['batch_size']

  data = torch.load(data_dir)

  train_val_data, test_data = train_test_split(data, test_size=0.1)
  train_data, val_data = train_test_split(train_val_data, train_size=0.7)

  dataset_params = dict(window_size=window_size, target_window_size=5)
  train_dataset = EthTransGraphDataset(train_data, **dataset_params)
  val_dataset = EthTransGraphDataset(val_data, **dataset_params)
  test_dataset = EthTransGraphDataset(test_data, **dataset_params)

  datamodule_params = dict(batch_size=batch_size, num_workers=1)
  datamodule = LightningDataset(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, **datamodule_params)

  model = ETHGT(config)

  return model, datamodule

def train(init_config, trainer_config):
  """Trains model with Pytorch Lightning Trainer"""
  init_config = torch.load(init_config)
  model, datamodule = init_model_dataset(init_config, data_dir=data_path)  
  log_n_steps = len(datamodule.train_dataset) // init_config['batch_size']
  trainer = pl.Trainer(log_every_n_steps=log_n_steps, **trainer_config)
  trainer.fit(model=model, datamodule=datamodule)
  return model, datamodule