
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