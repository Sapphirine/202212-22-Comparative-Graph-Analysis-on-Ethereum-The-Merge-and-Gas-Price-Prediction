import os 
import torch
import pytorch_lightning as pl 
from sklearn.model_selection import train_test_split
from torch_geometric.data.lightning.datamodule import LightningDataset
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.tune.integration.pytorch_lightning import Callback
from ray.tune.schedulers import PopulationBasedTraining

from .config import data_file_name
from .dataset import EthTransGraphDataset
from .model import ETHGT

data_path = os.path.join(os.getcwd(), data_file_name)

class TuneReportCallback(Callback):
  def on_validation_end(self, trainer, pl_module):
    tune.report(
      loss=trainer.callback_metrics["val_loss"].cpu() 
      )

def train_tune_checkpoint(config,
                                checkpoint_dir=None,
                                num_epochs=10,
                                num_gpus=0,
                                data_dir=data_path):
    """Training method for Ray Tune"""                                
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

    kwargs = {
        "max_epochs": num_epochs,
        'log_every_n_steps': 10,
        "gpus": num_gpus,
        "callbacks": [
            TuneReportCallback()
        ],
        "logger": TensorBoardLogger(
            save_dir=os.getcwd(), name="", version=".")
    }

    if checkpoint_dir:
        kwargs["resume_from_checkpoint"] = os.path.join(
            checkpoint_dir, "checkpoint")
    
    config['num_edge_features'] = train_dataset.num_edge_features
    config['num_features'] = train_dataset.num_features
    
    model = ETHGT(config)
    
    trainer = pl.Trainer(**kwargs)
    trainer.fit(model=model, datamodule=datamodule)

def tune_pbt(num_samples=10, num_epochs=10, gpus_per_trial=0, checkpoint_dir=None):
    """Tunes hyper-parameter with PBT method."""
    window_size = tune.choice([5, 10])
    batch_size = tune.choice([32, 64])
    lr = tune.loguniform(1e-5, 1e-2)
    dropout = tune.uniform(0.2, 0.2)
    hidden_size = tune.choice([8, 16, 32])
    in_heads = tune.choice([4, 8])
    out_size = tune.choice([16, 32, 64])
    out_heads = 1
    hidden_dim = tune.choice([16, 32, 64])
    num_heads = tune.choice([4, 8])
    num_layers = tune.choice([i for i in range(1, 5+1)])

    config = dict(
        lr=lr,
        dropout=dropout,
        hidden_size=hidden_size, 
        in_heads=in_heads, 
        out_size=out_size,
        out_heads=out_heads,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        window_size=window_size,
        batch_size=batch_size
        )

    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-4, 1e-1),
            'batch_size': [16, 32],
            'window_size': [5, 10, 20]
            })
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_tune_checkpoint,
                num_epochs=num_epochs,
                num_gpus=gpus_per_trial,
                checkpoint_dir=checkpoint_dir
                ),
            resources={
                "cpu": 4,
                "gpu": gpus_per_trial
            }
        ),

        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),

        run_config=air.RunConfig(
            name="tune_mnist_pbt",
        ),
        
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)   
    return results   
