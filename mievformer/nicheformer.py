from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
import math
from einops import rearrange, repeat, reduce
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from typing import Any, Optional
import pytorch_lightning as pl
from torch import nn
from torch import Tensor
from torch.nn import init
import numpy as np
from sklearn.neighbors import NearestNeighbors

class NicheDataSet(torch.utils.data.Dataset):
    def __init__(self, z: np.array, pos: np.array, neighbor_num: int = 100):
        """
        Arguments:
            z: np.array, shape ``[cell_num, latent_dim]``
            pos: np.array, shape ``[cell_num, space_dim]``
            neighbor_num: int, number of neighbors to consider
        store nearest neighors of z in self.zs
        store relative position of nearest neighbors in self.rel_pos
        """
        super().__init__()
        self.z = torch.tensor(z).float()
        self.input_dim = z.shape[-1]
        self.pos = torch.tensor(pos).float()
        self.neighbor_num = neighbor_num
        self.knn = NearestNeighbors(n_neighbors=neighbor_num)
        self.knn.fit(pos)
        self.neighbor_idx = self.knn.kneighbors(pos)[1]
    
    def __getitem__(self, index: int) -> Tensor:
        return self.z[self.neighbor_idx[index]], self.pos[self.neighbor_idx[index]] - self.pos[index]

    def __len__(self) -> int:
        return len(self.z)

class MultiNicheDataSet(torch.utils.data.Dataset):
    def __init__(self, z: np.array, pos: np.array, batchs: np.array, neighbor_num: int = 100):
        """
        Arguments:
            z: np.array, shape ``[cell_num, latent_dim]``
            pos: np.array, shape ``[cell_num, space_dim]``
            batch: np.array, shape ``[cell_num]``
            neighbor_num: int, number of neighbors to consider
        store nearest neighors of z in self.zs
        store relative position of nearest neighbors in self.rel_pos
        """
        super().__init__()
        self.batchs = batchs
        self.batch_uniq = batchs.unique()
        self.batch_nums = pd.Series(batchs).value_counts()[self.batch_uniq]
        expected_batchs = np.concatenate([np.repeat(batch, n) for batch, n in self.batch_nums.items()])
        try:
            assert np.all(expected_batchs == batchs)
        except:
            raise ValueError(f'batchs must be a continous in the order, {self.batch_uniq}.')
        self.ds_dict = {
            batch: NicheDataSet(z[batchs == batch], pos[batchs == batch], neighbor_num) for batch in self.batch_uniq
        }
        self.batch_start = {
            batch: self.batch_nums[:i].sum() for i, batch in enumerate(self.batch_uniq)
        }
        self.batch_one_hots = {
            batch: torch.eye(len(self.batch_uniq))[i] for i, batch in enumerate(self.batch_uniq)
        }
    
    def __getitem__(self, index: int) -> Tensor:
        batch = self.batchs[index]
        ds = self.ds_dict[batch]
        zs, relpos = ds[index - self.batch_start[batch]]
        batch_one_hot = self.batch_one_hots[batch]
        return zs, relpos, batch_one_hot

    def __len__(self) -> int:
        return self.batch_nums.sum()


class PositionalEncoding(nn.Module):
    """ Positional encoding module for the transformer """
    def __init__(self, d_model: int, space_dim: int = 2):
        super().__init__()
        coeffs = torch.exp(torch.arange(0, d_model, 2 * space_dim) * (-math.log(10000.0) / d_model))
        self.register_buffer('coeffs', coeffs)
        self.space_dim = space_dim
        self.p2h = nn.Sequential(
            nn.Linear(space_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model))
        
    def forward(self, pos: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        phases = pos.unsqueeze(-1) * self.coeffs
        phases = rearrange(phases, 'seq_len batch_size space_dim h_dim -> seq_len batch_size (space_dim h_dim)')
        pos_encoding = torch.cat([phases.sin(), phases.cos()], dim=-1)
        # pos_encoding = self.p2h(pos)
        return pos_encoding


class NicheEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, d_model: int = 64, space_dim=2, stochastic=True, head_num=1, num_layers=3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.pos_encoder = PositionalEncoding(d_model=d_model, space_dim=space_dim)
        print('head_num:', head_num)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=head_num, dim_feedforward=d_model)
        self.trnasformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.make_input_embedding = nn.Linear(input_dim, d_model)
        self.mask_embedding = nn.Embedding(1, input_dim)
        self.h2mu = nn.Linear(d_model, latent_dim)
        self.h2sigma = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.Softplus())
        self.stochastic = stochastic
        
    def forward(self, zs: Tensor, rel_pos: Tensor) -> Tensor:
        """
        Arguments:
            zs: Tensor, shape ``[batch_size, seq_len, input_dim]``
            rel_pos: Tensor, shape ``[batch_size, seq_len, seq_len, space_dim]``
        """
        batch_size, seq_len, input_dim = zs.shape
        zs = rearrange(zs, 'batch_size seq_len input_dim -> seq_len batch_size input_dim')
        rel_pos = rearrange(rel_pos, 'batch_size seq_len input_dim -> seq_len batch_size input_dim')
        zs[0] = self.mask_embedding(torch.zeros(batch_size, dtype=torch.long, device=zs.device))
        zs = self.make_input_embedding(zs)
        zs = zs + self.pos_encoder(rel_pos)
        zs = self.trnasformer_encoder(zs)[0]
        mu = self.h2mu(zs)
        sigma = self.h2sigma(zs)
        qe = torch.distributions.Normal(mu, sigma)
        if self.training:
            e = qe.rsample()
        else:
            e = qe.loc
        return e, qe


class DiffusionModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, h_dim: int = 128, beta_min: float=0.1, beta_max: float=20) -> None:
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.input2h = nn.Linear(input_dim, h_dim)
        self.t2h = PositionalEncoding(h_dim, space_dim=1)
        self.e2h = nn.Linear(latent_dim, h_dim)
        self.h2error = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_dim))

    def error_predictor(self, z: Tensor, e: Tensor, t: Tensor):
        h = self.input2h(z) + self.e2h(e) + self.t2h(t)
        return self.h2error(h)
        
    def purturb_z(self, z: Tensor, t: Tensor):
        mu_coeff = torch.exp(- 0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min)
        sigma_coeff = (1 - torch.exp(- 0.5 * t**2 * (self.beta_max - self.beta_min) - t * self.beta_min))**0.5
        eps = torch.randn_like(z)
        z = z * mu_coeff + sigma_coeff * eps
        return z, eps
    
    def time_coeff(self, t: Tensor) -> Tensor:
        return 1

    def loss_func(self, p: Tensor, eps: Tensor, t: float) -> Tensor:
        return torch.mean(self.time_coeff(t) * (p - eps)**2)
    
    def forward(self, z: Tensor, e: Tensor, t: float) -> Tensor:
        """
        Arguments:
            z: Tensor, shape ``[batch_size, input_dim]``
            e: Tensor, shape ``[batch_size, latent_dim]``
        """
        z, eps = self.purturb_z(z, t)
        p = self.error_predictor(z, e, t)
        p_loss = self.loss_func(p, eps, t)
        return p, eps, p_loss


class Distributor(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, h_dim: int = 128) -> None:
        super().__init__()
        self.e2w = nn.Sequential(
            nn.Linear(latent_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)
        )
        self.z2wb = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim + 1)
        )
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, e, z):
        w_e = self.e2w(e)
        wb_z = self.z2wb(z)
        w_z, b_z = wb_z[..., :-1], wb_z[..., -1]
        lp = self.logsoftmax(w_e @ w_z.T + b_z)
        return lp
        

class PlBaseModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        loss_dict = self.loss(*batch)
        loss = sum(loss_dict.values())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss_dict = self.loss(*batch)
        for key, val in loss_dict.items():
            self.log(f'val_{key}', val)
        loss = sum(loss_dict.values())
        self.log('val_loss', loss)
        return loss   
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
    



class PseudoEntropy(nn.Module):
    def __init__(self, nneighbor=8, dist_space='latent') -> None:
        super().__init__()
        self.nneighbor = nneighbor
        if dist_space not in ['latent', 'lp']:
            raise ValueError('dist_space must be one of "latent" or "lp"')
        self.dist_space = dist_space
    
    def forward(self, e, lp):
        edist = torch.cdist(e, e)
        if self.dist_space == 'latent':
            dist = edist
        elif self.dist_space == 'lp':
            dist = - torch.exp(lp) @ lp.T
        k_indices = torch.topk(dist, self.nneighbor, dim=-1, largest=False).indices
        knn_dist = edist[torch.arange(edist.size()[0])[:, None].expand(-1, self.nneighbor), k_indices]
        ref_std = (e.std(dim=0)**2).mean()
        pseudo_entropy = (knn_dist**2).mean() / ref_std
        return pseudo_entropy

        
        
class TotalKLD(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, e):
        mu_pe = torch.zeros(self.dim).to(e.device)
        cov_pe = torch.eye(self.dim).to(e.device)
        mu_e = e.mean(dim=0)
        cov_e = (e - mu_e).T @ (e - mu_e) / e.shape[0] + 1.0e-3 * cov_pe
        qe = torch.distributions.multivariate_normal.MultivariateNormal(mu_e, cov_e)
        pe = torch.distributions.multivariate_normal.MultivariateNormal(mu_pe, cov_pe)
        kld = torch.distributions.kl_divergence(qe, pe) / self.dim
        return kld
        


#  the NicheFormer model which is a subclass of the pytorch.lightning.LightningModule
# NicheFormer take latent cellstates and its neghbor cell states as input
# make a transformer embeddign from the latent cell states
class NicheFormer(PlBaseModule):
    def __init__(self, input_dim, latent_dim, train_ds, val_ds, kld_ld=0.1, pent_ld=0.05, dist_space='latent', batch_size=128, batch_correct=False, epoch_size=100000, num_layers=3, head_num=1) -> None:
        super().__init__()
        if batch_correct:
            batch_num = len(train_ds.batch_uniq)
            self.niche_encoder = NicheEncoder(input_dim=input_dim, latent_dim=latent_dim, num_layers=num_layers, head_num=head_num)
            self.distributor = Distributor(input_dim=input_dim, latent_dim=latent_dim + batch_num)
            self.forward = self.forward_multi_batch
        else:
            self.niche_encoder = NicheEncoder(input_dim=input_dim, latent_dim=latent_dim, stochastic=kld_ld > 0, num_layers=num_layers, head_num=head_num)
            self.distributor = Distributor(input_dim=input_dim, latent_dim=latent_dim)
            self.forward = self.forward_one_batch
        self.pseudo_entropy = PseudoEntropy(dist_space=dist_space)
        self.tot_kld = TotalKLD(latent_dim)
        self.kld_ld = kld_ld
        self.pent_ld = pent_ld
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.epoch_size = epoch_size

    def qe_kld(self, qe):
        pe = torch.distributions.Normal(torch.zeros_like(qe.loc), torch.ones_like(qe.scale))
        kld = torch.distributions.kl_divergence(qe, pe).sum(dim=-1)
        return kld

    def forward_multi_batch(self, *batch):
        zs, rel_pos, batchs = batch
        z = torch.clone(zs[:, 0])
        # zs = torch.cat([zs, repeat(batchs, 'b s -> b k s', k=zs.size()[1])], dim=-1)
        e, qe = self.niche_encoder(zs, rel_pos)
        eb = torch.cat([e, batchs], dim=-1)
        lp = self.distributor(eb, z)
        return e, qe, lp

    def forward_one_batch(self, *batch):
        zs, rel_pos, *others = batch
        z = torch.clone(zs[:, 0])
        e, qe = self.niche_encoder(zs, rel_pos)
        lp = self.distributor(e, z)
        return e, qe, lp
    
    # def output_wbs(self, *batch):
    def loss(self, *batch):
        e, qe, lp = self(*batch)
        kld = self.qe_kld(qe)
        # p_entropy = self.pseudo_entropy(e, lp)
        p_entropy = 0
        return {'reconst': torch.mean(- torch.diagonal(lp)), 'kld':self.kld_ld * torch.mean(kld), 'p_entropy': self.pent_ld * p_entropy}

    
    def metrics(self, *batch):
        loss_dict = self.loss(*batch)
        loss_dict['kld'] = loss_dict['kld'] / (self.kld_ld + 1.0e-10)
        loss_dict['p_entropy'] = loss_dict['p_entropy'] / (self.pent_ld + 1.0e-10)
        loss_dict = {key: vel.detach().cpu().item() for key, vel in loss_dict.items()}
        return loss_dict
    
    def train_dataloader(self):
        # default used by the Trainer
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            sampler=SubsetRandomSampler(
            torch.randint(high=len(self.train_ds), size=(self.epoch_size,)))
        )
        return train_loader


class FFlayer(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim=256, layer_num=2):
        super().__init__()
        layers = [
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim, elementwise_affine=False),
            nn.ReLU(inplace=True),
            
        ] + [
                nn.Sequential(
                    nn.Linear(mid_dim, mid_dim),
                    nn.LayerNorm(mid_dim, elementwise_affine=False),
                    nn.ReLU(inplace=True))
                for _ in range(layer_num - 1)
        ] + [nn.Linear(mid_dim, out_dim)]
        self.f = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.f(x)



def calc_nb_loss(ld, theta, obs):
    # ld = norm_mat * ld
    ld = ld + 1.0e-16
    theta = theta + 1.0e-16
    lp =  ld.log() - (theta).log()
    p_z = torch.distributions.NegativeBinomial(theta, logits=lp)
    l = - p_z.log_prob(obs)
    return(l)



class ScDistDataset(torch.utils.data.Dataset):
    def __init__(self, xsc, scnorm, xsp, spnorm, e, sample_size=10000) -> None:
        self.xsc = torch.tensor(xsc).float()
        self.scnorm = torch.tensor(scnorm).float()
        self.xsp = torch.tensor(xsp).float()
        self.spnorm = torch.tensor(spnorm).float()
        self.e = torch.tensor(e).float()
        self.sample_size = sample_size
        self.resample_pairs()
    
    def resample_pairs(self):
        self.sc_index = torch.randint(high=len(self.xsc), size=(self.sample_size,))
        self.sp_index = torch.randint(high=len(self.xsp), size=(self.sample_size,))
    
    def __getitem__(self, index):
        sci = self.sc_index[index]
        spi = self.sp_index[index]
        return self.xsc[sci], self.scnorm[sci], self.xsp[spi], self.spnorm[spi], self.e[spi]
    
    def __len__(self):
        return self.sample_size
        


class ScDistributor(PlBaseModule):
    def __init__(self, x_dim, z_dim, latent_dim, train_ds, batch_size=128) -> None:
        super().__init__()
        self.distributor = Distributor(input_dim=z_dim, latent_dim=latent_dim)
        self.x2z = FFlayer(x_dim, z_dim)
        self._scst_a = nn.Parameter(torch.zeros(x_dim))
        self.scst_b = nn.Parameter(torch.zeros(x_dim))
        self._sp_theta = nn.Parameter(torch.zeros(x_dim))
        self.softplus = nn.Softplus()
        self.train_ds = train_ds
        self.batch_size = batch_size
        self.reset_parameters()
    
    def reset_parameters(self):
        init.normal_(self._scst_a)
        init.normal_(self.scst_b)
        init.normal_(self._sp_theta)
    
    def sp_theta(self):
        return self.softplus(self._sp_theta)

    def scst_a(self):
        return self.softplus(self._scst_a)

    
    def forward(self, xsc, xnorm, e):
        z = self.x2z(xsc)
        p = torch.exp(self.distributor(e, z))
        xsp_hat = self.softplus(self.scst_a() * (p @ (xsc / (xnorm + 1.0e-10))) + self.scst_b)
        return xsp_hat, p
    
    def loss(self, xsc, scnorm, xsp, spnorm, e):
        xsp_hat, p = self(xsc, scnorm, e)
        loss_dict = {}
        try:
            loss_dict['nb_loss'] = calc_nb_loss(xsp_hat * spnorm, self.sp_theta(), xsp).sum(dim=-1).mean()
        except:
            import pdb;pdb.set_trace()
        return loss_dict  
    
    def load_optimized_distributor(self, distributor):
        self.distributor.load_state_dict(distributor.state_dict())
        for param in self.distributor.parameters():
            param.requires_grad = False
            
        
    def train_dataloader(self):
        self.train_ds.resample_pairs()
        # default used by the Trainer
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=True
        )
        return train_loader
    
    def impute_xsp(self, xsc, e, xsc_full, scnorm_full):
        z = self.x2z(xsc)
        p = torch.exp(self.distributor(e, z))
        xsp_hat_full = p @ (xsc_full / scnorm_full)
        return xsp_hat_full, p
        
        



class Encoder(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim, h_dim, enc_num_layers, enc_dist=torch.distributions.LogNormal): 
        super().__init__()
        ff_layers = [FFlayer(h_dim, h_dim) for _ in range(enc_num_layers)]
        self.enc_z_params = nn.Sequential(FFlayer(x_dim + c_dim, h_dim), *ff_layers, FFlayer(h_dim, 2 * z_dim))
        self.z_dim = z_dim
        self.soft_plus = nn.Softplus()
        self.enc_dist = enc_dist
        
    def forward(self, x, c):
        x = torch.cat([x, c], dim=-1)
        loc_rscale = self.enc_z_params(x)
        loc, scale = loc_rscale[..., :self.z_dim], loc_rscale[..., self.z_dim:]
        scale = self.soft_plus(scale)
        qz = self.enc_dist(loc, scale)
        if self.training:
            z = qz.rsample()
        else:
            z = loc
        return z, qz


# def calc_nb_loss(ld, theta, obs):
#     ld = ld + 1.0e-16
#     theta = theta + 1.0e-16
#     lp =  ld.log() - (theta).log()
#     p_z = torch.distributions.NegativeBinomial(theta, logits=lp)
#     l = - p_z.log_prob(obs)
#     return(l)


def calc_kld(qz):
    kld = -0.5 * (1 + qz.scale.pow(2).log() - qz.loc.pow(2) - qz.scale.pow(2))
    return(kld)


class Decoder(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim, h_dim, dec_num_layers):
        super().__init__()
        ff_layers = [FFlayer(h_dim, h_dim) for _ in range(dec_num_layers)]
        self.dec_x_mu = nn.Sequential(FFlayer(z_dim + c_dim, h_dim), *ff_layers, FFlayer(h_dim, x_dim))

    def reset_parameters(self):
        init.normal_(self._rtheta)
    
    def forward(self, z, c):
        z = torch.cat([z, c], dim=-1)
        x_mu = self.dec_x_mu(z)
        return x_mu



class scVAEDataSet(torch.utils.data.Dataset):
    def __init__(self, x, c):
        self.x = torch.tensor(x).float()
        self.c = torch.tensor(c).float()
        norm_mat = self.x.mean(axis=0) * self.x.mean(axis=1)[:, None]
        self.norm_mat = self.x.mean() * norm_mat / norm_mat.mean()
        
    def __getitem__(self, index):
        return self.x[index], self.norm_mat[index], self.c[index]
    
    def __len__(self):
        return len(self.x)


class scVAE(PlBaseModule):
    def __init__(self, x_dim, c_dim, z_dim, h_dim, enc_num_layers, dec_num_layers, train_ds, val_ds, batch_size=128) -> None:
        super().__init__()
        self.enc_z = Encoder(x_dim, c_dim, z_dim, h_dim, enc_num_layers)
        self.enc_l = Encoder(x_dim, c_dim, 1, h_dim, enc_num_layers, enc_dist=torch.distributions.LogNormal)
        self.dec_x = Decoder(x_dim, c_dim, z_dim, h_dim, dec_num_layers)
        self._rtheta = nn.Parameter(torch.zeros(x_dim))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.vamp_num = 128
        self.kld_ld = 1.0
        self.batch_size = batch_size
        self._vamp_w = nn.Parameter(torch.zeros(self.vamp_num))
        self._pz_u = nn.Parameter(torch.zeros(self.vamp_num, x_dim))
        pseudo_c = torch.eye(c_dim)[torch.randint(high=c_dim, size=(self.vamp_num,))].float()
        self.register_buffer('pseudo_c', pseudo_c)

    @property
    def theta(self):
        return self.softplus(self._rtheta)

    @property
    def pz_u(self):
        return self.softplus(self._pz_u)
    
    @property
    def lvamp_w(self):
        return self.logsoftmax(self._vamp_w)
        
    def forward(self, x, c):
        z, qz = self.enc_z(x, c)        
        l, ql = self.enc_l(x, c)        
        x_mu = self.dec_x(z, c)
        return z, qz, l, ql, x_mu
    
    def loss(self, x, norm_mat, c):
        z, qz, l, ql, x_mu  = self(x, c)
        loss_dict = {
            'lx': calc_nb_loss(norm_mat * x_mu * l, self.theta, x).sum(dim=-1).mean(),
            # 'kld_z': self.calc_z_kld_vamp(qz, c).mean() * self.kld_ld,
            'kld_z': calc_kld(qz).mean() * self.kld_ld,
            'kld_l': calc_kld(ql).sum(dim=-1).mean() * self.kld_ld
        }
        return loss_dict

    def train_dataloader(self):
        # default used by the Trainer
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            sampler=SubsetRandomSampler(
            torch.randint(high=len(self.train_ds), size=(10000,)))
        )
        return train_loader

    def log_pz_vamp(self, z, batch):
        uz, quz = self.enc_z(self.pz_u, self.pseudo_c)
        log_pz_k = self.lvamp_w + quz.log_prob(z.unsqueeze(-2)).sum(dim=-1) 
        log_pz = log_pz_k.logsumexp(dim=-1)
        return log_pz

    def calc_z_kld_vamp(self, qz, batch):
        # kld of pz and qz
        z = qz.rsample()
        log_pz = self.log_pz_vamp(z, batch)
        hqz = qz.entropy().sum(dim=-1)
        z_kld = - log_pz - hqz
        return z_kld

class MaskedLinearSig(nn.Module):
    def __init__(self, x_in, x_out, M, soft=False):
        super().__init__()
        self.lin = nn.Linear(x_in, x_out)
        self.sig = nn.Sigmoid()
        self.M = M
        self.soft = soft
    
    def forward(self, x):
        if self.soft:
            w = self.lin.weight
        else:
            w = self.lin.weight * self.M
        x = x @ w + self.lin.bias
        x = self.sig(x)
        return x

    @property    
    def l1_unmask_w(self):
        loss = (self.lin.weight * (1 - self.M)).abs()

class KineticsModel(nn.Module):
    def __init__(self, x_dim, lig_idxs, rec_idxs, tf_idxs, M_lig2rec, M_rec2tf, M_tf2tg) -> None:
        super().__init__()
        self.lig_idxs = lig_idxs
        self.rec_idxs = rec_idxs
        self.tf_idxs = tf_idxs
        self.lig2rec = MaskedLinearSig(lig_idxs.shape[0], rec_idxs.shape[0], M_lig2rec)
        self.rec2tf = MaskedLinearSig(rec_idxs.shape[0], tf_idxs.shape[0], M_rec2tf)
        self.tf2tg = MaskedLinearSig(tf_idxs.shape[0], x_dim, M_tf2tg, soft=True)
        self._degrads = nn.Parameter(x_dim)
        self._max_alphas = nn.Parameter(x_dim)
    
    def reset_parameters(self):
        init.normal_(self._degrads)
        init.normal_(self._max_alphas)
    
    @property
    def degrads(self):
        return self.exp(self._degrads)
    
    @property
    def max_alphas(self):
        return self.exp(self._max_alphas)
    
    def forward(self, x, p):
        nn_lig_exps = p @ x[..., self.lig_idxs]
        rec_act = self.lig2rec(nn_lig_exps)
        tf_act = self.rec2tf(rec_act * x[..., self.rec_idxs])
        tg_alpha = self.tf2tg(tf_act * x[..., self.tf_idxs]) * self.max_alphas
        l1_w_tf2tg = self.tf2tg.l1_unmask_w.mean()
        v_kinetics = tg_alpha - self.degrads * x
        return v_kinetics, l1_w_tf2tg



class NicheDynamics(PlBaseModule):
    def __init__(self, x_dim, c_dim, z_dim, h_dim, enc_num_layers, dec_num_layers):
        super().__init__()
        self.enc_z = Encoder(x_dim, c_dim, z_dim, h_dim, enc_num_layers)
        self.enc_l = Encoder(x_dim, c_dim, 1, h_dim, enc_num_layers, enc_dist=torch.distributions.LogNormal)
        self.dec_x = Decoder(x_dim, c_dim, z_dim, h_dim, dec_num_layers)
        self._rtheta = nn.Parameters(x_dim)
        self.softplus = nn.Softplus()
        self.ld_b = 1.0
        self.ld_m = 10.0
    
    def calc_opt_b(self, dp, dmupdz, p):
        opt_b = (dp + dmupdz) * p / (p**2 + self.ld_b)
        return opt_b
    
    def calc_opt_m(self, dp, dmupdz, p):
        opt_m = (dp + dmupdz) / (1 + self.ld_m)
        return opt_m
    
    def forward(self, e, z):
        with torch.no_grad():
            lp = self.distributor(e, z)
        p = torch.exp(lp)
        de = self.env_dyn(e, z)
        dp = jvp(p, e, de)
        mu = self.zdyn_in_env(e, z)
        dmupdz = divergence((mu * p), z, dim=-1)
        opt_b = self.calc_opt_b(dp, dmupdz, p)
        opt_m = self.calc_opt_m(dp, dmupdz, p)
        dp_hat = opt_b * p + opt_m - dmupdz
        dist_dyn_loss = (nn.MSELoss(dp, dp_hat) * p).mean()
        v_latent = jvp(self.dec_x, z, mu)
        x = self.dec_x(z)
        v_kinetics, l1_w_tf2tg = self.calc_v_kinetics(p, x)
        kinetics_loss = (nn.MSELoss(v_latent, v_kinetics) * p).mean()
        return dist_dyn_loss, kinetics_loss, l1_w_tf2tg


