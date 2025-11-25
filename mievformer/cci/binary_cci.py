import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch import nn
import torch
from torch import distributions as dist
from einops import einsum, reduce, rearrange, repeat
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau



class EncoderW(nn.Module):
    def __init__(self, celltype_dim, phi_dim, gene_dim):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(celltype_dim, phi_dim, gene_dim) * 0.1)
        self._sigma = nn.Parameter(torch.randn(celltype_dim, phi_dim, gene_dim) * 0.1 - 1)
    
    @property
    def sigma(self):
        return torch.exp(self._sigma)
    
    def forward(self):
        qw = dist.Normal(self.mu, self.sigma) 
        w = qw.rsample()
        return w, qw


def lognormalize_binary_lp(pre_lp):
    lp = F.log_softmax(torch.stack([pre_lp, torch.zeros_like(pre_lp)], dim=-1), dim=-1)
    lp = torch.clamp(lp, -20, -1e-3)
    return lp

def make_binary_pi(pre_lpi_s):
    pi_s = torch.sigmoid(pre_lpi_s)
    pi_s = torch.stack([pi_s, 1 - pi_s], dim=-1) + 1e-10
    return pi_s

class EncoderS(nn.Module):
    def __init__(self, e_dim, celltype_dim, hidden_dim, gene_dim, num_layers=2):
        super().__init__()
        self.ct_embeddings = nn.Embedding(celltype_dim, hidden_dim)
        self.e2h = nn.Linear(e_dim, hidden_dim)
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(2 * hidden_dim if len(layers) == 0 else hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, gene_dim))
        self.norm = nn.Sigmoid()
        self.h2lpi_s = nn.Sequential(*layers)

    def forward(self, e, k):
        h_ct = self.ct_embeddings(k)
        h_e = self.e2h(e)
        h = torch.cat([h_ct, h_e], dim=-1)
        pre_lpi_s = self.h2lpi_s(h)
        # pre_lpi_s = 10 * self.norm(pre_lpi_s)
        # import pdb;pdb.set_trace()
        pi_s = make_binary_pi(pre_lpi_s)
        lpi_s = torch.log(pi_s + 1e-10)
        qs = dist.Categorical(probs=pi_s)
        if self.training:
            s = F.gumbel_softmax(lpi_s, tau=0.1)
        else:
            s = F.gumbel_softmax(lpi_s, tau=0.1, hard=True)
        s = s[..., 0]
        return s, qs
    
class DecoderX(nn.Module):
    def __init__(self, gene_dim, x_mean):
        super().__init__()
        self.a = nn.Parameter(torch.randn(gene_dim))
        self.b = nn.Parameter(torch.randn(gene_dim))
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        norm_x_mean = x_mean / x_mean.sum()
        lnorm_x_mean = torch.log(norm_x_mean)
        self.register_buffer('x_coeff', lnorm_x_mean)
    
    def reset_parameters(self):
        nn.init.normal_(self.a)
        nn.init.normal_(self.b)
    
    def forward(self, s):
        lx_hat = self.logsoftmax(self.a * s + self.b + self.x_coeff)
        return lx_hat
    


class DecoderS(nn.Module):
    def __init__(self, e_dim, hidden_dim, num_layers=2, bias=-3.0, norm='batch'):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(e_dim if len(layers) == 0 else hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        if norm == 'layer':
            layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
        elif norm == 'batch':
            layers.append(nn.BatchNorm1d(hidden_dim))
        self.e2phi = nn.Sequential(*layers)
        self.bias = bias
    
    def monte_carlo_pi_s(self, e, ws):
        phi = self.e2phi(e)
        pre_lpi_s = torch.einsum('bd,skdg->bskg', phi, ws) + self.bias
        pi_s = reduce(make_binary_pi(pre_lpi_s)[..., 0], 'b s k g -> b k g', 'mean')
        return pi_s
        
    def forward(self, e, w_k):
        phi = self.e2phi(e)
        pre_lpi_s = torch.einsum('bd,bdg->bg', phi, w_k) + self.bias
        pi_s = make_binary_pi(pre_lpi_s)
        ps = dist.Categorical(probs=pi_s)
        return ps
        


class BinaryCCI(pl.LightningModule):
    def __init__(self, batch_size, celltype_dim, phi_dim, hidden_dim, gene_dim, e_dim, x_mean, num_layers=2):
        super().__init__()
        self.tot_batch_size = batch_size
        self.enc_w = EncoderW(celltype_dim, phi_dim, gene_dim)
        self.enc_s = EncoderS(e_dim, celltype_dim, hidden_dim, gene_dim, num_layers=num_layers)
        self.dec_x = DecoderX(gene_dim, x_mean)
        self.dec_s = DecoderS(e_dim, phi_dim, num_layers=num_layers)

    def forward(self, x, k, e):
        w, qw = self.enc_w()
        s, qs = self.enc_s(e, k)
        lx_hat = self.dec_x(s)
        ps = self.dec_s(e, w[k])
        return lx_hat, w, qw, s, qs, ps
    
    def fn_rec_loss(self, x, lx_hat):
        loss = - (x * lx_hat).sum(dim=1).mean()
        return loss
        
    def fn_kld_w(self, qw):
        pw = dist.Normal(0, 1)
        loss = dist.kl_divergence(qw, pw).sum() / self.tot_batch_size
        return loss
    
    def fn_kld_s(self, qs, ps):
        loss = dist.kl_divergence(qs, ps).sum(dim=1).mean()
        return loss

    def loss(self, x, k, e):
        lx_hat, w, qw, s, qs, ps = self(x, k, e)
        rec_loss = self.fn_rec_loss(x, lx_hat)
        kld_w = self.fn_kld_w(qw)
        kld_s = self.fn_kld_s(qs, ps)
        loss_dict = {
            'rec_loss': rec_loss
            # 'kld_w': kld_w,
            # 'kld_s': kld_s
        }
        return loss_dict

    def training_step(self, batch, batch_idx):
        x, k, e = batch
        loss_dict = self.loss(x, k, e)
        loss = sum(loss_dict.values())
        self.log_dict(loss_dict, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, k, e = batch
        loss_dict = self.loss(x, k, e)
        loss = sum(loss_dict.values())
        loss_dict['loss'] = loss
        vloss_dict = {f'val_{k}': v for k, v in loss_dict.items()}
        self.log_dict(vloss_dict, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.0e-3)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=5),
            "monitor": "val_loss",
        },
        }
    
    def variational_mean_pi_s(self, e):
        ks = repeat(torch.arange(self.enc_s.ct_embeddings.num_embeddings, device=e.device), 'k -> k b', b=e.size(0))
        logits_list = []
        for k in ks:
            s, qs = self.enc_s(e, k)
            logits_list.append(qs.logits[..., 0])
        logits = torch.stack(logits_list, dim=1)
        return torch.exp(logits)


    def monte_carlo_pi_s(self, e, n_samples=100):
        w, qw = self.enc_w()
        ws = qw.rsample((n_samples,))
        pi_s = self.dec_s.monte_carlo_pi_s(e, ws)
        return pi_s
        

class BinaryCCIDataset(torch.utils.data.Dataset):
    def __init__(self, data, celltype_labels, e_features):
        self.data = data
        self.celltype_labels = celltype_labels
        self.e_features = e_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        k = self.celltype_labels[idx]
        e = self.e_features[idx]
        return x, k, e


def make_BinaryCCI(adata, phi_dim=32, hidden_dim=256, celltype_label='celltype', e_feature_label='e', num_layers=2):
    # Extract data from AnnData object
    adata = adata.copy()
    if isinstance(adata.layers['counts'], np.ndarray):
        data = torch.tensor(adata.layers['counts'], dtype=torch.float32)
    else:
        data = torch.tensor(adata.layers['counts'].toarray(), dtype=torch.float32)
    # Determine dimensions
    celltype_dim = len(adata.obs[celltype_label].cat.categories)
    gene_dim = adata.shape[1]
    e_dim = adata.obsm[e_feature_label].shape[1]    
    # Initialize model
    x_mean = data.mean(dim=0)
    model = BinaryCCI(adata.shape[0], celltype_dim, phi_dim, hidden_dim, gene_dim, e_dim, x_mean, num_layers=num_layers)
    return model, data


def apply_binary_cci_to_adata(adata, max_epochs=100, batch_size=128, phi_dim=32, hidden_dim=256, celltype_label='celltype', e_feature_label='e', ckpt_path='checkpoints/', num_layers=2):
    model, data = make_BinaryCCI(adata, phi_dim, hidden_dim, celltype_label, e_feature_label, num_layers=num_layers)
    # Create dataset and dataloader
    celltype_labels = torch.tensor(adata.obs[celltype_label].cat.codes.values, dtype=torch.long)
    e_features = torch.tensor(adata.obsm[e_feature_label], dtype=torch.float32)
    dataset = BinaryCCIDataset(data, celltype_labels, e_features)
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders for training and validation
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize PyTorch Lightning trainer with early stopping and checkpointing
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, dirpath=ckpt_path, filename='best-checkpoint')
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[early_stopping, checkpoint_callback], devices=[0] if torch.cuda.is_available() else None)

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    # Load the best checkpoint after training
    best_model_path = checkpoint_callback.best_model_path
    celltype_dim = len(adata.obs[celltype_label].cat.categories)
    gene_dim = adata.shape[1]
    e_dim = adata.obsm[e_feature_label].shape[1]    
    x_mean = data.mean(dim=0)
    if best_model_path:
        model = BinaryCCI.load_from_checkpoint(
            best_model_path,
            batch_size=adata.shape[0],
            celltype_dim=celltype_dim,
            phi_dim=phi_dim,
            hidden_dim=hidden_dim,
            gene_dim=gene_dim,
            e_dim=e_dim,
            x_mean=x_mean,
            num_layers=num_layers
        )

    model.to('cpu')
    return model

def calculate_coactivate_probs(adata, model, lt_df, ligand_col='ligand', receptor_col='receptor', e_feature_label='e', celltype_label='celltype', device='cuda:0'):
    e_features = torch.tensor(adata.obsm[e_feature_label], dtype=torch.float32)
    celltype_labels = torch.tensor(adata.obs[celltype_label].cat.codes.values, dtype=torch.long)
    lt_df = lt_df.query(f'{ligand_col} in @adata.var_names and {receptor_col} in @adata.var_names')
    gene2idx = pd.Series(range(len(adata.var_names)), index=adata.var_names)
    nligands = lt_df[ligand_col].map(gene2idx).values
    nreceptors = lt_df[receptor_col].map(gene2idx).values

    # Create a DataLoader for e_features and celltype_labels
    ref_ct_vec = adata.uns['ref_adata_obs'][celltype_label]
    celltype_p_df = pd.DataFrame({
        celltype: np.exp(adata.obsm['dist'])[:, ref_ct_vec == celltype].sum(axis=1) 
        for celltype in adata.obs[celltype_label].cat.categories
    }, index=adata.obs_names)
    ct_p_tsr = torch.tensor(celltype_p_df.values, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(e_features, ct_p_tsr)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()
    model.to(device)
    coactivate_probs_sum = 0
    total_samples = 0
    pi_s_list = []
    with torch.no_grad():
        for batch_e_features, ct_p in dataloader:
            batch_e_features = batch_e_features.to(device)
            ct_p = ct_p.to(device)
            pi_s = model.variational_mean_pi_s(batch_e_features)
            # pi_s = model.monte_carlo_pi_s(batch_e_features)
            pi_s = pi_s * ct_p.unsqueeze(-1)
            pi_s_list.append(pi_s)
            ligand_probs = rearrange(pi_s[..., nligands], 'b k g -> b k 1 g')
            receptor_probs = rearrange(pi_s[..., nreceptors], 'b k g -> b 1 k g')
            coactivate_prob = (ligand_probs * receptor_probs)
            coactivate_probs_sum += coactivate_prob.sum(dim=0)
            total_samples += batch_e_features.size(0)
    mean_coactivate_prob_tsr = coactivate_probs_sum / total_samples
    # Concatenate pi_s and store in adata.layers for each cell type
    pi_s_all = torch.cat(pi_s_list, dim=0)
    for i, celltype in enumerate(adata.obs[celltype_label].cat.categories):
        adata.layers[f'pi_s_{celltype}'] = pi_s_all[:, i, :].cpu().numpy()
    # Convert the tensor to a DataFrame using pivot
    coactivate_prob_df = pd.DataFrame(mean_coactivate_prob_tsr.cpu().numpy().reshape(-1, mean_coactivate_prob_tsr.shape[-1]), 
                                        columns=[f'{row[ligand_col]}_{row[receptor_col]}' for _, row in lt_df.iterrows()])
    coactivate_prob_df['sender'] = np.repeat(adata.obs[celltype_label].cat.categories, mean_coactivate_prob_tsr.shape[1])
    coactivate_prob_df['receiver'] = np.tile(adata.obs[celltype_label].cat.categories, mean_coactivate_prob_tsr.shape[0])
    coactivate_prob_df = coactivate_prob_df.melt(id_vars=['sender', 'receiver'], var_name='ligand_receptor', value_name='coactivate_prob')
    coactivate_prob_df[['ligand', 'receptor']] = coactivate_prob_df['ligand_receptor'].str.split('_', expand=True)
    coactivate_prob_df = coactivate_prob_df.drop(columns=['ligand_receptor'])
    return coactivate_prob_df, adata
