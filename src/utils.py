import torch
import numpy as np
import pandas as pd 
import plotly.express as px 

def plot_predict_curve_helper(model, datamodule,
                              window: int = 5, 
                              scaler_file_name=None):
  pred_ys = []
  ys = []
  with torch.no_grad():
    for x, y in iter(datamodule.test_dataloader()):
      pred_y = model.forward(x)
      pred_ys.append(pred_y.cpu().numpy())
      ys.append(y.cpu().numpy())

  pred_ys = np.vstack(pred_ys).flatten()  
  ys = np.hstack(ys) 
  if scaler_file_name != None:
    scaler = torch.load(scaler_file_name) 
    ys = ys * scaler.scale_[2] + scaler.mean_[2]
    pred_ys = pred_ys * scaler.scale_[2] + scaler.mean_[2]

  df = pd.DataFrame([moving_average(ys, window), moving_average(pred_ys, window)],
                    index=['true', 'predict']).T   
  return df 


def plot_predict_curve(*args, **kwargs):
  df = plot_predict_curve_helper(*args, **kwargs)
  fig = px.line(data_frame=df)
  return fig 


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
    