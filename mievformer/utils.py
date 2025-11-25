import math
import scanpy as sc
import squidpy as sq
import pandas as pd
from scipy.stats import norm
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.spatial.distance import pdist, squareform


def split_dataset(ds, test_ratio, val_ratio):
    n = len(ds)
    n_test = int(test_ratio * n)
    n_val = int(val_ratio * n)
    n_train = n - n_test - n_val
    ds_train, ds_val, ds_test = torch.utils.data.random_split(ds, [n_train, n_val, n_test])
    return ds_train, ds_val, ds_test

@torch.no_grad()
def output_metrics(ds, model, batch_size=128):
    data_loader = DataLoader(ds, batch_size=batch_size, num_workers=12, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    metric_names = ['reconst', 'kld', 'p_entropy']
    metric_list_dict = {
        metric: [] for metric in metric_names
    }
    for batch in data_loader:
        zs, res_pos = batch
        with torch.no_grad():
            loss_dict = model.metrics(zs.to(device), res_pos.to(device))
        for metric in metric_names:
            metric_list_dict[metric].append(loss_dict[metric])
    mean_metric_dict = {
        metric: np.mean(metric_list_dict[metric]) for metric in metric_names
    }
    model.to('cpu')
    return mean_metric_dict


@torch.no_grad()
def output_niche_rep(ds, model, batch_size=128):
    data_loader = DataLoader(ds, batch_size=batch_size, num_workers=12, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    e_list = []
    for batch in data_loader:
        batch = [el.to(device) for el in batch]
        with torch.no_grad():
            e, qe, lp = model(*batch)
        e_list.append(e.detach().cpu())
    e = torch.cat(e_list, dim=0)
    model.to('cpu')
    return e


@torch.no_grad()
def output_wbs(ds, model, batch_size=128):
    data_loader = DataLoader(ds, batch_size=batch_size, num_workers=12, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    w_e_list = []
    w_z_list = []
    b_z_list = []
    for batch in data_loader:
        e, z = batch
        w_e = model.distributor.e2w(e.to(device))
        wb_z = model.distributor.z2wb(z.to(device))
        w_z, b_z = wb_z[..., :-1], wb_z[..., -1]
        w_e_list.append(w_e.detach().cpu())
        w_z_list.append(w_z.detach().cpu())
        b_z_list.append(b_z.detach().cpu())
    model.to('cpu')
    w_e = torch.cat(w_e_list, dim=0)
    w_z = torch.cat(w_z_list, dim=0)
    b_z = torch.cat(b_z_list, dim=0)
    return w_e, w_z, b_z

@torch.no_grad()
def output_dist_refs(ds, model, ref_z, batch_size=128):
    data_loader = DataLoader(ds, batch_size=batch_size, num_workers=12, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dist_list = []
    for batch in data_loader:
        e,  = batch
        dist = model.distributor(e.to(device), ref_z.to(device))
        dist_list.append(dist.detach().cpu())
    dist = torch.cat(dist_list, dim=0)
    model.to('cpu')
    return dist

@torch.no_grad()
def output_scdist(ds, xsc, xsc_full, model, batch_size=128):
    data_loader = DataLoader(ds, batch_size=batch_size, num_workers=12, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    xsc = xsc.to(device)
    xsc_full = xsc_full.to(device)
    scnorm_full = calc_norm_mat(xsc_full)
    xsphat_list = []
    for e in data_loader:
        e = e[0].to(device)
        xsphat, scp = model.impute_xsp(xsc, e, xsc_full, scnorm_full)
        xsphat_list.append(xsphat.detach().cpu())
    xsphat = torch.cat(xsphat_list, dim=0).numpy()
    return xsphat

@torch.no_grad()
def output_celldist_pnmfae(ds, model, batch_size=128):
    data_loader = DataLoader(ds, batch_size=batch_size, num_workers=12, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_W = model.W.cpu().detach().numpy()
    model.to(device)
    model.eval()
    p_list = []
    alpha_list = []
    for batch in data_loader:
        cell_dist = batch
        with torch.no_grad():
            p, qp, est_cell_dist = model(cell_dist.to(device))
        p_list.append(p.detach().cpu().numpy())
        # alpha_list.append(qp.mean.detach().cpu().numpy())
    total_p = np.concatenate(p_list, axis=0)
    # total_alpha = np.concatenate(alpha_list, axis=0)
    model.to('cpu')
    return cell_W, total_p, total_p

@torch.no_grad()
def calculate_dist_diff(ds, model, batch_size=128, n_repeat=30):
    data_loader = DataLoader(ds, batch_size=batch_size, num_workers=12, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dist_list = []
    source_z_list = []
    target_z_list = []
    total_source_idx_list = []
    diff_list = []
    batch_idx = 0
    for batch in data_loader:
        zs, rel_pos = batch
        batch_idx_list = torch.arange(batch_idx, batch_idx + len(zs))
        batch_idx += len(zs)
        z = zs[:, 0]
        target_z_list.append(z.detach().cpu())
        e, qe = model.nniche_encoder(zs.to(device), rel_pos.to(device))
        lp = model.distributor(e, z.to(device))
        lz_orig = torch.diag(lp)
        batch_size = len(zs)
        lz_alt = 0
        source_idx_list = torch.randint(zs.shape[1], (batch_size, ))
        absolute_source_idx_list = ds.neighbor_idx[batch_idx_list, source_idx_list]
        total_source_idx_list.append(absolute_source_idx_list)
        source_z_list.append(zs[torch.arange(batch_size), source_idx_list].detach().cpu())
        for _ in range(n_repeat):
            zs_alt = zs.clone()
            swap_idx_list = torch.randint(len(z), (batch_size, ))
            swap_z = z[swap_idx_list]
            zs_alt[torch.arange(batch_size), source_idx_list] = swap_z
            e, qe = model.nniche_encoder(zs_alt.to(device), rel_pos.to(device))
            dist = model.distributor(e, z.to(device))
            lz_alt += torch.diag(dist)
        lz_alt /= n_repeat
        diff_list.append((lz_orig - lz_alt).detach().cpu())
    model.to('cpu')
    source_z = torch.cat(source_z_list, dim=0)
    target_z = torch.cat(target_z_list, dim=0)
    lz_diff = torch.cat(diff_list, dim=0)
    total_source_idx = np.concatenate(total_source_idx_list, axis=0)
    return source_z, target_z, lz_diff, torch.tensor(total_source_idx)


def output_scvae_z(ds, model, batch_size=128):
    data_loader = DataLoader(ds, batch_size=batch_size, num_workers=12, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    z_list = []
    for batch in data_loader:
        x, norm_mat, c = batch
        z, qz, l, ql, x_mu = model(x.to(device), c.to(device))
        z_list.append(qz.loc.detach().cpu())
    z = torch.cat(z_list, dim=0)
    model.to('cpu')
    return z


def subset_adata(adata, n_cells, random=True):
    if random:
        adata = adata[np.random.choice(adata.obs_names, n_cells, replace=False)]
    else:
        adata = adata[:n_cells]
    return adata


def spatial_subset_adata(adata, min_x, max_x, min_y, max_y):
    adata = adata[(adata.obsm['spatial'][:, 0] >= min_x) & (adata.obsm['spatial'][:, 0] <= max_x) &
                  (adata.obsm['spatial'][:, 1] >= min_y) & (adata.obsm['spatial'][:, 1] <= max_y)]
    return adata

def sample_next_cells(adata):
    tr_mat = adata.obsp['T_fwd']
    tr_mat = tr_mat / tr_mat.sum(axis=1)
    next_cells = [np.random.choice(adata.obs_names, p=tr_mat[i].A.flatten()) for i in np.arange(adata.shape[0])]
    return next_cells

def pairwise_pearson_corr(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    X = X / X.std(axis=0)
    Y = Y / Y.std(axis=0)
    corr = X.T @ Y / X.shape[0]
    return corr

def gaussian_base_p_values(vals):
    p_vals = norm.sf(vals, loc=np.mean(vals), scale=np.std(vals))
    return p_vals

def add_obsm_to_obs(adata, obsm_key):
    obsm = adata.obsm[obsm_key]
    if isinstance(obsm, pd.DataFrame):
        obsm_df = obsm
    else:
        obsm_df = pd.DataFrame(obsm, index=adata.obs_names)
    obsm_df.columns = [f'{obsm_key}_{col}' for col in obsm_df.columns]
    adata.obs[obsm_df.columns] = obsm_df
    return adata, obsm_df.columns.tolist()

def normalize_spatial(adata, neighbor_num, spatial_key='spatial'):
    pos = adata.obsm[spatial_key]
    knn = NearestNeighbors(n_neighbors=neighbor_num)
    knn.fit(pos)
    dists = knn.kneighbors(pos)[0]
    ref_dist = dists[:, -1].mean()
    adata.obsm[spatial_key] = adata.obsm[spatial_key] / ref_dist
    return adata

def get_center_region(adata, exp_plot_num=10000):
    mid_x, mid_y = np.median(adata.obsm['spatial'], axis=0)
    width_x, width_y = adata.obsm['spatial'].max(axis=0) - adata.obsm['spatial'].min(axis=0)
    if exp_plot_num > adata.shape[0]:
        width = np.max([width_x, width_y])
    else:
        coeff = np.sqrt(exp_plot_num / adata.shape[0])
        width = np.mean([width_x * coeff, width_y * coeff])
    sub_adata = spatial_subset_adata(adata, mid_x - width, mid_x + width, mid_y - width, mid_y + width)
    return sub_adata

def align_adata(sc_adata, sp_adata):
    common_genes = np.intersect1d(sc_adata.var_names, sp_adata.var_names)
    sc_adata = sc_adata[:, common_genes]
    sp_adata = sp_adata[:, common_genes]
    return sc_adata, sp_adata

def calc_norm_mat(x):
    norm_mat = x.sum(axis=1, keepdims=True) * x.sum(axis=0, keepdims=True)
    norm_mat = norm_mat / norm_mat.mean()
    return norm_mat
    

def sparse_std(x, axis=0):
    mean = np.array(x.mean(axis=axis)).reshape(-1)
    var = np.sqrt((np.array((x.multiply(x)).mean(axis=axis)).reshape(-1) - (mean * mean)))
    return var

def get_centroid_idx(x):
    dist_mat = squareform(pdist(x))
    mean_dist = dist_mat.mean(axis=0)
    idx = np.argmin(mean_dist)
    return idx


def get_clsuters_centroid_idxs(adata, cluster_key, rep_key='e', ref_num=1000):
    cluster_ids = adata.obs[cluster_key].unique()
    cluster_centroid_idxs = {}
    for cluster_id in cluster_ids:
        cluster_adata = adata[adata.obs[cluster_key] == cluster_id]
        if cluster_adata.shape[0] > ref_num:
            cluster_adata = subset_adata(cluster_adata, ref_num)
        centroid_idx = get_centroid_idx(cluster_adata.obsm[rep_key])
        cluster_centroid_idxs[cluster_id] = cluster_adata.obs_names[centroid_idx]
    return cluster_centroid_idxs

@torch.no_grad()
def output_dist_params(ds, model, batch_size=128):
    data_loader = DataLoader(ds, batch_size=batch_size, num_workers=12, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    e_list = []
    mu_list = []
    sigma_list = []
    for batch in data_loader:
        batch = [el.to(device) for el in batch]
        with torch.no_grad():
            e, qe, lp = model(*batch)
        e_list.append(e.detach().cpu())
        mu_list.append(qe.loc.detach().cpu())
        sigma_list.append(qe.scale.detach().cpu())
    e = torch.cat(e_list, dim=0)
    mu = torch.cat(mu_list, dim=0)
    sigma = torch.cat(sigma_list, dim=0)
    model.to('cpu')
    return e, mu, sigma