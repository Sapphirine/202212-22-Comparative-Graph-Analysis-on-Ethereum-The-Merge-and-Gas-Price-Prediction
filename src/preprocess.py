import torch 
import ray
import pandas as pd 
from functools import partial
from sklearn.preprocessing import StandardScaler

from .config import preprocess_config, numeric_features
from .embedding import gen_data


def parallel_gen_data(agg_obj, data_params,
                      gen_data=gen_data,
                      preprocess_config=preprocess_config):
  """Parallel generates graph data with Ray
  
  Args:
    agg_obj: pd.groupby
    data_params: dict, parameters for generating data
    preprocess_config: dict, parameters for ray init
  
  Returns:
    data: pd.DataFrame[Pytorch_geometric Data], index is timestep
  """
  gen_data = partial(gen_data, **data_params)

  # parallel
  # A day with 720 objects, each train 5 epochs on 8 cpus -> 2h!
  # 110MB+
  #
  
  ray.init(num_cpus=preprocess_config['num_cpus'], 
            num_gpus=preprocess_config['num_gpus']
            )

  @ray.remote
  def pargen_data(kv, gen_data_func):
      return [kv[0], gen_data_func(kv[1])]
  pargen_data = partial(pargen_data.remote, gen_data_func=gen_data)
  
  res_ids = [pargen_data(kv) for kv in agg_obj]
  res = ray.get(res_ids)

  ray.shutdown()
  data = pd.DataFrame(res, columns=['timestep', 'data']).set_index('timestep')
  return data


def scale_numeric_features(df, 
                           numeric_features=numeric_features, 
                           scaler=StandardScaler()):
  """Scales numerical features in df"""
  scaler.fit(df[numeric_features])
  df.loc[:, numeric_features] = scaler.transform(df[numeric_features])
  return df, scaler