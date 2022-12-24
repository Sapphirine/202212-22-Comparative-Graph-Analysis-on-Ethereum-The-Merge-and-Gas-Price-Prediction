import torch 
import torch.nn as nn
import torch.nn.functional as F 
import pytorch_lightning as pl 
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, global_mean_pool

class AttentionBlock(nn.Module):
    def __init__(self, **kwargs):
        """A Pytorch implementation of Transformer Attention block.

        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(**kwargs['ln_params'])
        self.attn = nn.MultiheadAttention(**kwargs['attn_params'])
        self.layer_norm_2 = nn.LayerNorm(**kwargs['ln_params'])
        self.linear = nn.Sequential(
            nn.Linear(**kwargs['inlinear_params']),
            nn.GELU(),
            nn.Dropout(**kwargs['dropout']),
            nn.Linear(**kwargs['outlinear_params']),
            nn.Dropout(**kwargs['dropout']),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        causal_mask = torch.tril(torch.ones((inp_x.shape[1], inp_x.shape[1]))).to(x.device)
        x = x + self.attn(inp_x, inp_x, inp_x, attn_mask=causal_mask)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

class CausalTrans(torch.nn.Module):
  """Transformer Encoder with causal mask"""
  def __init__(self, **kwargs):
    super().__init__()
    num_layers = 3 if 'num_layers' not in kwargs else kwargs['num_layers']
    self.transformer = nn.Sequential(
            *(AttentionBlock(**kwargs['attnblock_params']) for _ in range(num_layers))
        )
  def forward(self, x):
    x = self.transformer(x)
    return x 

class ETHGT(pl.LightningModule):
  """Proposed model, extracts graph features with GATConv,
     which are then pass through Causal Transformer"""
  def __init__(self, config):
    super().__init__()

    lr = config['lr']
    dropout = config['dropout']
    hidden_size = config['hidden_size']
    in_heads = config['in_heads']
    out_size = config['out_size']
    out_heads = config['out_heads']
    embed_dim = config['out_size']
    hidden_dim = config['hidden_dim']
    num_heads = config['num_heads']
    window_size = config['window_size']
    num_edge_features = config['num_edge_features']
    num_features = config['num_features']
    num_layers = config['num_layers']

    attnblock_params = {
        'ln_params': {'normalized_shape': embed_dim},
        'attn_params': {'embed_dim': embed_dim, 'num_heads': num_heads, 'batch_first': True},
        'inlinear_params': {'in_features': embed_dim, 'out_features': hidden_dim},
        'outlinear_params': {'in_features': hidden_dim, 'out_features': embed_dim}, 
        'dropout': {'p': dropout}
    }
    transformer_params = {
        'attnblock_params': attnblock_params,
        'num_layers': num_layers
    }
    kwargs = {
        'GAT_params_1': {
            'edge_dim': num_edge_features,
            'in_channels': num_features, 
            'out_channels': hidden_size,
            'heads': in_heads,
            'dropout': dropout
            },
        'GAT_params_2': {
            'edge_dim': num_edge_features,
            'in_channels': hidden_size * in_heads, 
            'out_channels': out_size,
            'heads': out_heads,
            'dropout': dropout
            },
        'transformer_params': transformer_params,
        'regressor_params': {'in_features': embed_dim, 'out_features': 1},
        'conv_forward_params': {
            'return_attention_weights': False
        },
        'reshape_params': (window_size, -1),
        'lr': lr
    }

    self.conv1 = GATv2Conv(**kwargs['GAT_params_1'])
    self.conv2 = GATv2Conv(**kwargs['GAT_params_2'])
    self.causal_transformer = CausalTrans(**kwargs['transformer_params'])
    self.linear = Linear(**kwargs['regressor_params'])
    self.init_params = kwargs
  
  def forward(self, b):
    x, edge_index, edge_attr, batch = (
        b.x, b.edge_index, b.edge_attr, b.batch) 
    o = self.conv1(x, edge_index, edge_attr, **self.init_params['conv_forward_params'])
    x = o if type(o) != tuple else o[0]  # attention weights
    x = F.relu(x)
    o = self.conv2(x, edge_index, edge_attr, **self.init_params['conv_forward_params'])
    x = o if type(o) != tuple else o[0]  # log to explore
    x = F.relu(x)

    # readout layer
    x = global_mean_pool(x, batch)
    
    # Temporal layer
    window_size, vec_size = self.init_params['reshape_params']
    temporal_batch = (batch[-1].item() + 1) // window_size
    x = x.reshape(temporal_batch, window_size, vec_size)
    x = self.causal_transformer(x)

    # regression layer
    x = x[:, -1, :]
    x = self.linear(x)
    x = torch.tanh(x)  # tanh to (-1, 1)
    return x

  def mse_loss(self, pred, true):
    return F.mse_loss(pred, true)
  
  def training_step(self, train_batch, batch_idx):
    x, y = train_batch 
    preds = self.forward(x)
    loss = self.mse_loss(preds.flatten(), y)
    self.log('train_loss', loss)
    return loss 

  def validation_step(self, val_batch, batch_idx):
    x, y = val_batch
    preds = self.forward(x)
    loss = self.mse_loss(preds.flatten(), y)
    return {'val_loss': loss}
  
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    self.log("val_loss", avg_loss)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.init_params['lr'])
    return optimizer 