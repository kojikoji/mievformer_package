from torch.utils.data import DataLoader, SubsetRandomSampler
from .nicheformer import PlBaseModule
from einops import rearrange, repeat, reduce
import torch
import pytorch_lightning as pl
from torch import nn
from torch import Tensor
import numpy as np
from torch import distributions as D


class DistDataset(torch.utils.data.Dataset):
    def __init__(self, cell_dist):
        self.cell_dist = cell_dist

    def __len__(self):
        return len(self.cell_dist)

    def __getitem__(self, idx):
        return self.cell_dist[idx]


class AlphaEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.h2alpha = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Softmax(dim=-1)
        )
        self.h2beta = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus()
        )
    
    def forward(self, cell_dist):
        h = nn.functional.softplus(self.fc(cell_dist))
        alpha_gamm = self.h2alpha(h)
        beta_gamm = self.h2beta(h)
        # qp = D.gamma.Gamma(alpha_gamm, beta_gamm)
        # alpha = qp.rsample()
        return alpha_gamm, alpha_gamm


class ProbNMFAE(PlBaseModule):
    def __init__(self, input_dim, latent_dim, train_ds, val_ds, batch_size=128, w_ld=1.0):
        super().__init__()
        self.enc_alpha = AlphaEncoder(input_dim, latent_dim)
        self._prior_alpha_gamm = nn.Parameter(torch.zeros(latent_dim))
        self._prior_beta_gamm = nn.Parameter(torch.zeros(latent_dim))
        self._raw_W = nn.Parameter(torch.randn(latent_dim, input_dim))
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.w_ld = w_ld
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self._prior_alpha_gamm)
        nn.init.normal_(self._prior_beta_gamm)
        nn.init.normal_(self._raw_W)
    
    @property
    def W(self):
        return nn.functional.softmax(self._raw_W, dim=-1)
    
    @property
    def prior_alpha_gamm(self):
        return nn.functional.softplus(self._prior_alpha_gamm)
    
    @property
    def prior_beta_gamm(self):
        return nn.functional.softplus(self._prior_beta_gamm)
    
    def forward(self, cell_dist):
        alpha, qp = self.enc_alpha(cell_dist)
        est_cell_dist = alpha @ self.W
        return alpha, qp, est_cell_dist
    
    def loss(self, cell_dist):
        alpha, qp, est_cell_dist = self(cell_dist)
        loss_dict = {}
        loss_dict['reconst'] = - (cell_dist * torch.log(est_cell_dist + 1e-10)).sum(dim=-1).mean()
        loss_dict['glasso_w'] = self.w_ld * torch.norm(alpha, p=2, dim=0).mean()
        # loss_dict['kld'] = D.kl.kl_divergence(qp, D.gamma.Gamma(self.prior_alpha_gamm, self.prior_beta_gamm)).mean()
        return loss_dict
    
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            sampler=SubsetRandomSampler(
            torch.randint(high=len(self.train_ds), size=(10000,)))
        )
        return train_loader

    def training_step(self, batch, batch_idx):
        loss_dict = self.loss(batch)
        loss = sum(loss_dict.values())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss_dict = self.loss(batch)
        for key, val in loss_dict.items():
            self.log(f'val_{key}', val)
        loss = sum(loss_dict.values())
        self.log('val_loss', loss)
        return loss   
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
    
    