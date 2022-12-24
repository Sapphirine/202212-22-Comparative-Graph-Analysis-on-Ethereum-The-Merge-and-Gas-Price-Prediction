"""config file for experiments"""
import torch

data_file_name = 'test_data.pt'
model_name = 'ETHGT.pt'
scaler_file_name = 'test_data_scaler.pt'
project_id = 'bda-6893'
bucket_name = 'bda_eth_19g'

############
# init train 
############
init_config_name = 'init_config.pt'

############
# preprocess config
############
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

numeric_features = ['value', 'gas', 'gas_price', 'receipt_cumulative_gas_used',
                    'receipt_gas_used', 'transaction_type']
preprocess_config = {
  'num_cpus': 4,
  'num_gpus': 1,

}

edge_attr_cols = ['value', 'gas', 'gas_price', 'receipt_cumulative_gas_used', 'receipt_gas_used', 'transaction_type']
target = 'gas_price'
node2vec_params = {
      'embedding_dim': 16,
      'walk_length': 20,
      'context_size': 10,
      'walks_per_node': 20,
      'num_negative_samples': 1,
      'p': 200,
      'q': 1,
      'sparse': True
  }
loader_params = dict(batch_size=128, shuffle=True, num_workers=2)
optim_params = {'lr': 0.01}
loop_params = {'num_episodes': 5}
train_params = {'loader_params': loader_params, 'optim_params': optim_params, 'loop_params': loop_params}

data_params = {
    'x': {'node2vec_params': node2vec_params, 'train_params': train_params, 'device': device},
    'y': {'target': target},
    'edge': {'edge_attr_cols': edge_attr_cols}
    }

############
# tune config
############
tune_config = {
  'num_samples': 1,
  'num_epochs': 100,
  'gpus_per_trial': 1
}

############
# trainer config
############
trainer_config = {
  'max_epochs': 500,
  'accelerator': 'gpu',
  'devices': 1
}