import torch
import numpy as np
import pandas as pd 
from functools import partial
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data

from .config import device

def gen_data(v, **kwargs):
  """Generates pyg Data instance for each DataFrame"""
  data = Data()
  """target"""
  def _y(v, target):
    """Gets target column min in aggerated object(a time window)"""
    return v[target].min()
  data.y = _y(v, **kwargs['y'])

  """edge index | edge attr | addr idx"""
  def _edge(v, edge_attr_cols):
    addresses = np.array(
        list(set(np.hstack([v.from_address.values, v.to_address.values])))
        )
    addr_idx, addresses = pd.factorize(addresses)
    lookup_table = dict(zip(addresses, addr_idx))
    edge_index = np.array([v.from_address.map(lambda ad: lookup_table[ad]).values,
                        v.to_address.map(lambda ad: lookup_table[ad]).values])
    edge_index = torch.LongTensor(edge_index)
    edge_attr = torch.FloatTensor(v[edge_attr_cols].values.astype('float'))
    num_nodes = len(addr_idx)
    return edge_index, edge_attr, num_nodes
  data.edge_index, data.edge_attr, data.num_nodes = _edge(v, **kwargs['edge'])
   
  """node features"""  
  def gen_node_features(data, device, node2vec_params, train_params):
    def node2vec_func(edge_index, device, **kwargs):
      node2vec = Node2Vec(edge_index, **kwargs).to(device)
      return node2vec
    
    def train_func(node2vec, num_nodes, device, **kwargs):
      loader = node2vec.loader(**kwargs['loader_params'])
      optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), **kwargs['optim_params'])
      """train"""
      def _train(num_episodes):
        node2vec.train()
        for e in range(num_episodes):
          total_loss = 0
          for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
          print(f'Epoch: {e}, Loss: {total_loss / len(loader)}')
      _train(**kwargs['loop_params'])
      """result embeddings"""
      with torch.no_grad():
        node2vec.eval()
        x = node2vec(torch.arange(num_nodes, device=device))
      return x

    node2vec_func = partial(node2vec_func, **node2vec_params)
    train_func = partial(train_func, **train_params)
    node2vec = node2vec_func(data.edge_index, device)  
    x = train_func(node2vec, data.num_nodes, device)
    return x

  data.x = gen_node_features(data, **kwargs['x'])

  return data 

