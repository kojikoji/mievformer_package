import math
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Mapping
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from statsmodels.stats import multitest
from scipy import stats, linalg
from scipy.special import logsumexp, log_softmax, softmax
import torch
import torch.nn as nn
import pandas as pd
import scanpy as sc
import anndata
import numpy as np
from . import nicheformer as nf
from . import prob_nmfae as pnf
from . import utils
from .utils import output_dist_params
from . import sample_conditional_ca as scca
from sklearn.neighbors import NearestNeighbors


MIEVFORMER_RAW_E_KEY = 'mievformer_raw_e'
MIEVFORMER_CA_KEY = 'reference_probability_ca'
MIEVFORMER_DEFAULT_REPRESENTATION_KEY = 'mievformer_default_representation'
MIEVFORMER_SINGLE_SLICE_CA = 'reference_probability_ca'
MIEVFORMER_MULTI_SLICE_CA = 'sample_conditional_reference_probability_ca'


def _parse_bool_or_auto(value, parameter_name):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == 'auto':
        return 'auto'
    if normalized in {'1', 'true', 'yes'}:
        return True
    if normalized in {'0', 'false', 'no'}:
        return False
    raise ValueError(
        f"{parameter_name} must be true, false, or auto; got {value!r}"
    )


def resolve_mievformer_batch_correction(adata, batch_key=None, batch_correct='auto'):
    """Resolve batch correction, enabling it for data with multiple slices."""
    requested = _parse_bool_or_auto(batch_correct, 'batch_correct')
    if batch_key == 'same':
        batch_key = None
    if batch_key is None:
        n_samples = 1
    else:
        if batch_key not in adata.obs:
            raise KeyError(f"obs[{batch_key!r}] not found")
        values = pd.Series(adata.obs[batch_key], index=adata.obs_names)
        if values.isna().any():
            raise ValueError(f"obs[{batch_key!r}] contains missing sample labels")
        n_samples = int(values.astype(str).nunique())
        if n_samples < 1:
            raise ValueError(f"obs[{batch_key!r}] does not contain any samples")
    if requested == 'auto':
        return n_samples > 1
    if requested and n_samples == 1:
        raise ValueError(
            'Batch correction requires a batch_key with at least two samples'
        )
    return requested


def _mievformer_raw_e_key(adata, e_key=None):
    if e_key is not None:
        key = str(e_key)
    else:
        metadata = adata.uns.get(MIEVFORMER_DEFAULT_REPRESENTATION_KEY, {})
        if isinstance(metadata, Mapping) and metadata.get('raw_embedding_key'):
            key = str(metadata['raw_embedding_key'])
        elif MIEVFORMER_RAW_E_KEY in adata.obsm:
            key = MIEVFORMER_RAW_E_KEY
        else:
            key = 'e'
    if key not in adata.obsm:
        raise ValueError(f"obsm[{key!r}] is required as the raw Mievformer embedding")
    values = np.asarray(adata.obsm[key])
    if values.ndim != 2 or values.shape[0] != adata.n_obs:
        raise ValueError(f"obsm[{key!r}] must have one two-dimensional row per cell")
    if not np.isfinite(values).all():
        raise ValueError(f"obsm[{key!r}] contains non-finite values")
    return key


def _mievformer_raw_e(adata, e_key=None):
    return adata.obsm[_mievformer_raw_e_key(adata, e_key=e_key)]


@contextmanager
def _module_on_device(module, device):
    """Temporarily move a module to the CA device and restore it afterwards."""
    parameters = list(module.parameters())
    original_device = parameters[0].device if parameters else torch.device('cpu')
    was_training = module.training
    resolved_device = _reference_probability_ca_device(device)
    module.to(resolved_device)
    module.eval()
    try:
        yield resolved_device
    finally:
        module.to(original_device)
        module.train(was_training)



def adata2ds(adata, batch_key=None, neighbor_num=100):
    if batch_key is None:
        ds = nf.NicheDataSet(adata.obsm['nf_cellrep'], adata.obsm['spatial'], neighbor_num=neighbor_num)
    else:
        ds = nf.MultiNicheDataSet(adata.obsm['nf_cellrep'], adata.obsm['spatial'], adata.obs[batch_key].values, neighbor_num=neighbor_num)
    return ds
    

def analyze_state_dependence(adata, model, batch_key=None, neighbor_num=100):
    ds = adata2ds(adata, batch_key=batch_key, neighbor_num=neighbor_num)
    srouce_z, target_z, lz_diff, source_idx_list = utils.output_dist_refs(ds, model)
    pair_z = torch.cat([srouce_z, target_z], dim=1).numpy()
    adata.obsm['pair_z'] = pair_z
    adata.obs['lz_diff'] = lz_diff.numpy()
    adata.obsm['source_cell_id'] = adata.obs_names[source_idx_list.numpy()]
    source_obs_df = adata[source_idx_list.numpy()].obs
    source_obs_df.columns = [f'source_{col}' for col in source_obs_df.columns]
    adata.obs[source_obs_df.columns] = source_obs_df
    adata.obsm['X_eumap'] = adata.obsm['X_umap']
    

def add_dist_across_cells(
    adata,
    model,
    output_mode='original',
    ref_num=1000,
    e_key=None,
):
    raw_e = _mievformer_raw_e(adata, e_key=e_key)
    if 'batch_one_hot' in adata.obsm.keys():
        eb_mat = np.concatenate([raw_e, adata.obsm['batch_one_hot']], axis=1)
        eds = torch.utils.data.TensorDataset(torch.tensor(eb_mat).float())
    else:
        eds = torch.utils.data.TensorDataset(torch.tensor(raw_e).float())
    if adata.shape[0] > ref_num:
        ref_adata = utils.subset_adata(adata, ref_num)
    else:
        ref_adata = adata.copy()
    ref_z = torch.tensor(ref_adata.obsm['nf_cellrep']).float()
    dist = utils.output_dist_refs(eds, model, ref_z)
    if output_mode == 'original':
        adata.obsm['dist'] = dist.numpy()
        adata.uns['ref_adata_obs'] = ref_adata.obs
        ref_adata.obsm['dist'] = dist.numpy().T
        return adata, ref_adata
    elif output_mode == 'dadata':
        adata = anndata.AnnData(dist.numpy(), obs=adata.obs, var=ref_adata.obs, uns=adata.uns)
        return adata
    else:
        raise('output_mode must be one of original or dadata')

def add_wb_ez(adata, model, cell_rep_key='X_pca', e_key=None):
    raw_e = _mievformer_raw_e(adata, e_key=e_key)
    if 'batch_one_hot' in adata.obsm.keys():
        eb_mat = np.concatenate([raw_e, adata.obsm['batch_one_hot']], axis=1)
        ezds = torch.utils.data.TensorDataset(torch.tensor(eb_mat).float(), torch.tensor(adata.obsm[cell_rep_key]).float())
    else:
        ezds = torch.utils.data.TensorDataset(torch.tensor(raw_e).float(), torch.tensor(adata.obsm[cell_rep_key]).float())
    w_e, w_z, b_z = utils.output_wbs(ezds, model)
    adata.obsm['w_e'] = w_e.numpy()
    adata.obsm['w_z'] = w_z.numpy()
    adata.obsm['b_z'] = b_z.numpy()
    return adata
    

# should be normalized for cell abundance across niches
def calculate_niche_specificity(adata, niche_cluster_key='leiden_e', ref_num=1000):
    niche_centroid_cells = utils.get_clsuters_centroid_idxs(adata, niche_cluster_key, ref_num=ref_num)
    for niche_cluster in adata.obs[niche_cluster_key].unique():
        niche_centroid_cell = niche_centroid_cells[niche_cluster]
        w_e = adata[niche_centroid_cell].obsm['w_e'].copy().flatten()
        adata.obs[f'niche_specificity_{niche_cluster}'] = ((adata.obsm['w_z'] * w_e).sum(axis=1) + adata.obsm['b_z'].flatten())
    return adata
    


def analyze_fate_determinant_cells(adata, model, alpha=0.1):
    adata.obs_names_make_unique()
    adata, ref_adata = add_dist_across_cells(adata, model)
    next_cells = utils.sample_next_cells(adata)
    next_adata = adata[next_cells].copy()
    for var_id in ['fate', 'dist']:
        adata.obsm[f'delta_{var_id}'] = pd.DataFrame(next_adata.obsm[f'{var_id}'].values - adata.obsm[f'{var_id}'].values, index=adata.obs_names, columns=adata.obsm[f'{var_id}'].columns)
    fate_corr_df = pd.DataFrame(utils.pairwise_pearson_corr(adata.obsm['delta_dist'].values, adata.obsm['delta_fate'].values), index=adata.obsm['delta_dist'].columns, columns=adata.obsm['delta_fate'].columns)
    ref_adata.obsm['fate_corr'] = fate_corr_df
    pval_mat = np.apply_along_axis(utils.gaussian_base_p_values, 0, fate_corr_df.values)
    ref_adata.obsm['fate_corr_pval'] = pd.DataFrame(pval_mat, index=fate_corr_df.index, columns=fate_corr_df.columns)
    flatten_pvals = pval_mat.flatten()
    qval_mat = multitest.multipletests(flatten_pvals, method='fdr_bh')[1].reshape(pval_mat.shape)
    ref_adata.obsm['fate_corr_qval'] = pd.DataFrame(qval_mat, index=fate_corr_df.index, columns=fate_corr_df.columns)
    ref_adata.obsm['fate_corr_l10qval'] = np.log10(ref_adata.obsm['fate_corr_qval'])
    term_states = adata.obsm['fate'].columns
    sig_corr_cell_df = pd.DataFrame({
        term_state:
       (ref_adata.obsm['fate_corr'][term_state] > ref_adata.obsm['fate_corr'][term_state].quantile(1 - alpha)).astype(str) 
       for term_state in term_states}, index=ref_adata.obs_names)
    ref_adata.obsm['fate_corr_sig'] = sig_corr_cell_df
    return adata, ref_adata

def embed_distribution(adata, model):
    adata.obs_names_make_unique()
    adata, ref_adata = add_dist_across_cells(adata, model)
    dadata = anndata.AnnData(adata.obsm['dist'].values, obs=adata.obs, var=ref_adata.obs, uns=adata.uns)
    dadata.raw = adata.raw
    sc.pp.scale(dadata, max_value=10)
    sc.pp.neighbors(dadata, use_rep='X', n_neighbors=100)
    sc.tl.umap(dadata)
    sc.tl.leiden(dadata, key_added='leiden_dist')
    return dadata

def optimize_prob_nmfae(adata, log_dir, max_epochs=1000, val_prop=0.1, ngpu=1, batch_size=128, ldim=20):
    ds = pnf.DistDataset(torch.nn.functional.softmax(torch.tensor(adata.obsm['dist']).float(), dim=-1))
    if len(ds) * val_prop < 1024:
        train_ds, val_ds = torch.utils.data.random_split(ds, [1.0 - val_prop, val_prop])
    else:
        train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - 1024, 1024])
    checkpoint_callback = ModelCheckpoint(dirpath=f'{log_dir}/ckpt')
    logger = TensorBoardLogger(save_dir=log_dir, version=1, name=log_dir)
    trainer = pl.Trainer(max_epochs=max_epochs,devices=ngpu, accelerator="gpu", callbacks=[EarlyStopping(monitor="val_loss", patience=20), checkpoint_callback], reload_dataloaders_every_n_epochs=1, strategy='ddp_find_unused_parameters_true', logger=logger)
    model = pnf.ProbNMFAE(input_dim=adata.obsm['dist'].shape[1], latent_dim=ldim, train_ds=train_ds, val_ds=val_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, pin_memory=True, drop_last=True)
    trainer.fit(model, val_dataloaders=val_loader)
    model = pnf.ProbNMFAE.load_from_checkpoint(checkpoint_callback.best_model_path, input_dim=adata.obsm['dist'].shape[1], latent_dim=ldim, train_ds=train_ds, val_ds=val_ds)
    return model


def ifinteger(x):
    if x is not np.ndarray:
        x = x.toarray()
    return np.all(x == x.astype(int))



def scale_adata(adata):
    adata = adata.copy()
    if ifinteger(adata.X):
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    sc.pp.scale(adata) 
    return adata


def exclude_common_targets_ligands(lt_df, q=0.99):
    select_props = (lt_df.values > lt_df.quantile(q, axis=0).values).astype(int).mean(axis=1)
    lt_df = lt_df.loc[select_props < 0.5, :]
    return lt_df

def normalize_lt_df(lt_df, genes, q=0.99):
    common_targets = np.intersect1d(lt_df.index.astype(str), genes.astype(str))
    common_ligands = np.intersect1d(lt_df.columns.astype(str), genes.astype(str))
    lt_df = lt_df.loc[common_targets, common_ligands]
    lt_df = exclude_common_targets_ligands(lt_df, q=q)
    norm_lt_df = pd.DataFrame((lt_df.values > lt_df.quantile(q, axis=0).values).astype(int), index=lt_df.index, columns=lt_df.columns)
    return norm_lt_df



def calculate_bf(diff_df, cluster_counts, sig_delta=1.0):
    total_coutns = cluster_counts.sum()
    uniq_n_delta = 1.0 / (1.0 / cluster_counts + 1.0 / (total_coutns - cluster_counts))
    nu = total_coutns - 2
    t_stats = diff_df['scores'].values
    n_deltas = uniq_n_delta[diff_df.group.values].values
    one_p_nsig = 1 + n_deltas * sig_delta**2
    bf = np.sign(t_stats) * ((nu + 1) / 2) * np.log(
        (1 + t_stats ** 2 / nu) / (1 + t_stats**2 / (nu * one_p_nsig))
        ) - 0.5 * np.log(one_p_nsig)
    return bf




def make_diff_df_niche_cell(adata, cell_label, niche_label='leiden_e'):
    diff_df_list = []
    cell_clusters = adata.obs[cell_label].unique()
    for cell_cluster in cell_clusters:
        cadata = adata[adata.obs[cell_label] == cell_cluster]
        if 'log1p' in cadata.uns and not 'base' in cadata.uns['log1p'].keys():
            cadata.uns['log1p']['base'] = None
        cluster_counts = cadata.obs[niche_label].value_counts()
        cadata = cadata[cadata.obs[niche_label].isin(cluster_counts[cluster_counts > 1].index)]
        try:
            sc.tl.rank_genes_groups(cadata, groupby=niche_label, method='t-test', use_raw=False)
        except:
            import pdb;pdb.set_trace()
        diff_df = sc.get.rank_genes_groups_df(cadata, group=None).assign(cell_cluster=cell_cluster)
        diff_df['bf'] = calculate_bf(diff_df, cadata.obs[niche_label].value_counts())
        diff_df['diff_prob'] = 1 / (1 + np.exp(-diff_df['bf']))
        diff_df_list.append(diff_df)
    diff_df = pd.concat(diff_df_list)
    diff_df['pvals_adj'] = multitest.multipletests(diff_df.pvals.values, method='fdr_bh')[1]
    return diff_df

    


def comm_prob_in_niche(lig_diff_df, act_diff_df):
    comm_diff_df = pd.merge(lig_diff_df, act_diff_df, on=['names', 'group'], suffixes=('_lig', '_act'))
    comm_diff_df['comm_prob'] = comm_diff_df['diff_prob_lig'] * comm_diff_df['diff_prob_act']
    return comm_diff_df

def comm_prob_in_niche_lr(lig_diff_df, rec_diff_df, lr_df):
    lig_diff_df = lig_diff_df.rename(columns={'names': 'ligand'})
    rec_diff_df = rec_diff_df.rename(columns={'names': 'receptor'})
    lig_diff_df = pd.merge(lr_df, lig_diff_df, on=['ligand'], suffixes=('', '_lig'))
    rec_diff_df = pd.merge(lr_df, rec_diff_df, on=['receptor'], suffixes=('', '_rec'))
    comm_diff_df = pd.merge(lig_diff_df, rec_diff_df, on=['ligand', 'receptor', 'group'], suffixes=('_lig', '_rec'))
    comm_diff_df['comm_prob'] = comm_diff_df['diff_prob_lig'] * comm_diff_df['diff_prob_rec']
    return comm_diff_df 
    

def lognoncom_prob(cci_df, bf1='bf_lig', bf2='bf_act'):
    from scipy import special
    zeros = np.zeros(cci_df.shape[0])
    lse_0_bfa_bflig = special.logsumexp(np.vstack([zeros, cci_df[bf2].values, cci_df[bf1].values]), axis=0)
    lse_0_bfa = special.logsumexp(np.vstack([zeros, cci_df[bf2].values]), axis=0)
    lse_0_bflig = special.logsumexp(np.vstack([zeros, cci_df[bf1].values]), axis=0)
    lnoncom_prob = lse_0_bfa_bflig - lse_0_bfa - lse_0_bflig
    return lnoncom_prob



def estimate_cci(adata, lt_df, cell_label, niche_label='leiden_e'):
    norm_lt_df = normalize_lt_df(lt_df, adata.var_names)
    lig_adata = adata[:, norm_lt_df.columns]
    target_adata = adata[:, norm_lt_df.index]
    var_means = np.mean(target_adata.X,axis=0)
    var_stds = utils.sparse_std(target_adata.X, axis=0)
    expect_size = 10000
    split_num = target_adata.shape[0] // expect_size + 1
    chunks = np.array_split(np.arange(target_adata.shape[0]), split_num)
    act_mat_list = []
    for chunk in chunks:
        act_mat_list.append(((target_adata.X[chunk].toarray() - var_means) / (var_stds + 1.0e-10)) @ norm_lt_df.values) 
    act_mat = np.array(np.concatenate(act_mat_list, axis=0))
    act_adata = anndata.AnnData(act_mat, obs=target_adata.obs, var=lig_adata.var)
    lig_diff_df = make_diff_df_niche_cell(lig_adata, cell_label, niche_label=niche_label)
    act_diff_df = make_diff_df_niche_cell(act_adata, cell_label, niche_label=niche_label)
    comm_diff_df = comm_prob_in_niche(lig_diff_df, act_diff_df)
    # comm_diff_df['comm_pval'] = np.vstack([comm_diff_df['pvals_adj_lig'].values, comm_diff_df['pvals_adj_act'].values]).max(axis=0)
    comm_diff_df = comm_diff_df.sort_values('comm_prob', ascending=False)
    vals = -lognoncom_prob(comm_diff_df)
    comm_diff_df.loc[:, 'nlncp' ] = vals
    comm_diff_df = comm_diff_df[['group','names', 'scores_lig','cell_cluster_lig','bf_lig','diff_prob_lig', 'scores_act','cell_cluster_act','bf_act','diff_prob_act','comm_prob', 'nlncp']]
    return comm_diff_df


def estimate_cci_lr(adata, lr_df, cell_label, niche_label='leiden_e', ligand_label='ligand_gene_symbol', receptor_label='receptor_gene_symbol'):
    lr_df = lr_df[[ligand_label,receptor_label]].drop_duplicates()
    lr_df.columns = ['ligand', 'receptor']
    uniq_ligadns = np.intersect1d(np.unique(lr_df['ligand'].values), adata.var_names)
    uniq_receptors = np.intersect1d(np.unique(lr_df['receptor'].values), adata.var_names)
    lig_adata = adata[:, uniq_ligadns]
    rec_adata = adata[:, uniq_receptors]
    lig_diff_df = make_diff_df_niche_cell(lig_adata, cell_label, niche_label=niche_label)
    rec_diff_df = make_diff_df_niche_cell(rec_adata, cell_label, niche_label=niche_label)
    comm_diff_df = comm_prob_in_niche_lr(lig_diff_df, rec_diff_df, lr_df)
    comm_diff_df = comm_diff_df.sort_values('comm_prob', ascending=False)
    comm_diff_df.loc[:, 'nlncp'] = -lognoncom_prob(comm_diff_df, bf1='bf_lig', bf2='bf_rec')
    return comm_diff_df
    
    

def clip_center_adata(adata, center_region_ratio=0.1):
    mid_x, mid_y = np.median(adata.obsm['spatial'], axis=0)
    width_x, width_y = adata.obsm['spatial'].max(axis=0) - adata.obsm['spatial'].min(axis=0)
    width_ratio = math.sqrt(center_region_ratio)
    center_width = np.mean([width_x * width_ratio, width_y * width_ratio])
    center_adata = utils.spatial_subset_adata(adata, mid_x - 0.5 * center_width, mid_x + 0.5 * center_width, mid_y - 0.5 * center_width, mid_y + 0.5 * center_width)
    return center_adata

def optimize_scdistributor(sc_adata, sp_adata, nf_model, z_dim, log_dir, max_epochs=1000, val_prop=0.1, ngpu=1, batch_size=128, val_gene_prop=0.1):
    e_dim = sp_adata.obsm['e'].shape[1]
    common_genes = np.intersect1d(sc_adata.var_names, sp_adata.var_names)
    train_genes = np.random.choice(common_genes, int(common_genes.shape[0] * (1 - val_gene_prop)), replace=False)
    val_genes = np.intersect1d(train_genes, common_genes)
    tsc_adata = sc_adata[:, train_genes]
    tsp_adata = sp_adata[:, train_genes]
    xsc = tsc_adata.layers['count'].toarray()
    xsp = tsp_adata.layers['counts'].toarray()
    scnorm = utils.calc_norm_mat(xsc)
    spnorm = utils.calc_norm_mat(xsp)
    e = sp_adata.obsm['e']
    sc_train_idx = np.random.choice(sc_adata.shape[0], int(sc_adata.shape[0] * (1 - val_prop)), replace=False)
    sc_val_idx = np.setdiff1d(np.arange(sc_adata.shape[0]), sc_train_idx)
    sp_train_idx = np.random.choice(sp_adata.shape[0], int(sp_adata.shape[0] * (1 - val_prop)), replace=False)
    sp_val_idx = np.setdiff1d(np.arange(sp_adata.shape[0]), sp_train_idx)
    train_ds = nf.ScDistDataset(
        xsc[sc_train_idx], scnorm[sc_train_idx], xsp[sp_train_idx], spnorm[sp_train_idx], e[sp_train_idx]
    )
    val_ds = nf.ScDistDataset(
        xsc[sc_val_idx], scnorm[sc_val_idx], xsp[sp_val_idx], spnorm[sp_val_idx], e[sp_val_idx]
    )
    checkpoint_callback = ModelCheckpoint(dirpath=f'{log_dir}/ckpt')
    logger = TensorBoardLogger(save_dir=log_dir, version=1, name=log_dir)
    trainer = pl.Trainer(max_epochs=max_epochs,devices=ngpu, accelerator="gpu", callbacks=[EarlyStopping(monitor="val_loss", patience=20), checkpoint_callback], reload_dataloaders_every_n_epochs=1, strategy='ddp_find_unused_parameters_true', logger=logger)
    model = nf.ScDistributor(train_genes.shape[0], z_dim, e_dim, train_ds, batch_size=batch_size)
    model.load_optimized_distributor(nf_model.distributor)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, pin_memory=True, drop_last=True)
    trainer.fit(model, val_dataloaders=val_loader)
    model = nf.ScDistributor.load_from_checkpoint(checkpoint_callback.best_model_path, x_dim=train_genes.shape[0], z_dim=z_dim, latent_dim=e_dim, train_ds=train_ds)
    return model, train_genes, val_genes

def distribute_sc_in_niche(sc_adata, sp_adata, model, train_genes, val_genes, batch_size=128, sc_ref_num=10000):
    sub_sc_index = np.random.choice(sc_adata.shape[0], sc_ref_num, replace=False)
    psc_adata = sc_adata[sub_sc_index, train_genes]
    xsc = torch.tensor(psc_adata.layers['count'].toarray()).float()
    xsc_full = torch.tensor(sc_adata[sub_sc_index].layers['count'].toarray()).float()
    eds = torch.utils.data.TensorDataset(torch.tensor(sp_adata.obsm['e']).float())
    model.eval()
    xsp_full = utils.output_scdist(eds, xsc, xsc_full, model, batch_size=batch_size)
    sp_imp_adata = anndata.AnnData(xsp_full, obs=sp_adata.obs, obsm=sp_adata.obsm, var=sc_adata.var, uns=sp_adata.uns)
    sub_sp_idx = np.random.choice(sp_adata.shape[0], 1000, replace=False)
    vsp_adata = sp_adata[sub_sp_idx, val_genes]
    vsp_imp_adata = sp_imp_adata[sub_sp_idx, val_genes]
    vcorrs = np.array([
        stats.pearsonr(vsp_adata.X.toarray()[:, i], vsp_imp_adata.X[:, i])[0] for i in np.arange(vsp_adata.shape[1])
    ])
    tsp_adata = sp_adata[sub_sp_idx, train_genes]
    tsp_imp_adata = sp_imp_adata[sub_sp_idx, train_genes]
    tcorrs = np.array([
        stats.pearsonr(tsp_adata.X.toarray()[:, i], tsp_imp_adata.X[:, i])[0] for i in np.arange(tsp_adata.shape[1])
    ])
    return sp_imp_adata, vcorrs, tcorrs

def visualize_cci_in_niche(cci_df, niche_cluster, senders=['Tumor_cell'], min_comm_prob=0.999, max_comm_num=30):
    cci_df = cci_df.dropna()
    cci_df = cci_df.query('cell_cluster_lig in @senders')
    cci_df = cci_df.query('not cell_cluster_act in @senders')
    cci_df = cci_df[cci_df['group'] == niche_cluster]
    cci_df = cci_df.query('cell_cluster_lig != cell_cluster_act')
    cci_df = cci_df.query(f'comm_prob > {min_comm_prob}')
    if cci_df.shape[0] > max_comm_num:
        cci_df = cci_df.sort_values('nlncp', ascending=False).iloc[:max_comm_num]
    uniq_celltypes = np.unique(np.concatenate([cci_df['cell_cluster_act'].values, cci_df['cell_cluster_lig'].values]))
    # colors = pd.Series(target_color_dict)[labels]
    import plotly.graph_objects as go
    uniq_ligands = np.unique(cci_df['names'].values)
    ligand_pos_dict = pd.Series({
        ligand: i
        for i, ligand in enumerate(uniq_ligands)
    })
    celltype_pos_dict = pd.Series({
        celltype: i + uniq_ligands.shape[0]
        for i, celltype in enumerate(uniq_celltypes)
    })
    senders = cci_df.cell_cluster_lig.values
    receivers = cci_df.cell_cluster_act.values
    ligands = cci_df.names.values
    sources = pd.concat([ligand_pos_dict.loc[ligands], celltype_pos_dict.loc[senders]]).values
    targets = pd.concat([celltype_pos_dict.loc[receivers], ligand_pos_dict.loc[ligands]]).values
    values = np.log(pd.concat([cci_df['nlncp'], cci_df['nlncp']])).values
    labels = np.concatenate([senders, senders])
    tot_list = np.concatenate([uniq_ligands, uniq_celltypes])
    fig = go.Figure(data=[go.Sankey(node=dict(label=tot_list),
        link=dict(
            source=sources,
            target=targets,
            value=values))])
    fig.update_layout(font_family="Courier New")
    return fig

def train_nicheformer(
    adata,
    model_params,
    train_params,
    model_save_path,
    log_dir_base
):
    """
    Train a NicheFormer model with the given parameters.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    model_params : dict
        Dictionary containing model parameters:
        - ngpu: Number of GPUs to use (default: 1)
        - bsize: Batch size (default: 512)
        - ldim: Latent dimension (default: 20)
        - klld: KL divergence loss weight (default: 0)
        - etld: Entropy loss weight (default: 0)
        - nlayers: Number of transformer layers (default: 3)
        - nheads: Number of attention heads (default: 1)
        - dsp: Distance space ('latent' or other) (default: 'latent')
        - crkey: Cell representation key (default: 'X_pca')
        - bkey: Batch key (default: None)
        - bcorr: Batch correction flag (default: False)
        - nn: Number of neighbors (default: 100)
    train_params : dict
        Dictionary containing training parameters:
        - max_epochs: Maximum number of epochs (default: 1000)
        - num_workers: Number of workers for data loading (default: 12)
    model_save_path : str
        Path to save the trained model.
    log_dir_base : str
        Base directory for saving logs.
    
    Returns
    -------
    tuple
        (trained_model, model_construct_params)
    """
    import os
    import torch
    import numpy as np
    import anndata
    import pytorch_lightning as pl
    from datetime import datetime
    from torch.utils.data import DataLoader
    from sklearn.neighbors import NearestNeighbors
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    import importlib
    
    # Import required custom modules (assuming they are available in the environment)
    # import nicheformer.workflow as wl  # Custom data processing module
    
    # Set default parameters if not provided
    model_params_defaults = {
        'ngpu': 1,
        'bsize': 512,
        'mxep': 1000,
        'nn': 100,
        'ldim': 20,
        'klld': 0,
        'etld': 0,
        'nlayers': 3,
        'nheads': 1,
        'dsp': 'latent',
        'crkey': 'X_pca',
        'bkey': None,
        'bcorr': 'false'
    }
    
    # Apply defaults for any missing parameters
    for key, default_value in model_params_defaults.items():
        if key not in model_params:
            model_params[key] = default_value
    
    # Extract parameters
    ngpu = int(model_params.get('ngpu', 1))
    batch_size = int(model_params.get('bsize', 512))
    max_epochs = int(model_params.get('mxep', 1000))
    if 'max_epochs' in train_params:
        max_epochs = int(train_params['max_epochs'])
    neighbor_num = int(model_params.get('nn', 100))
    latent_dim = int(model_params.get('ldim', 20))
    kld_ld = float(model_params.get('klld', 0))
    pent_ld = float(model_params.get('etld', 0))
    nlayers = int(model_params.get('nlayers', 3))
    nheads = int(model_params.get('nheads', 1))
    dspace = model_params.get('dsp', 'latent')
    cellrep_key = model_params.get('crkey', 'X_pca')
    batch_key = model_params.get('bkey', None)
    batch_correct = model_params.get('bcorr', 'false') == 'true'
    num_workers = train_params.get('num_workers', 12)
    
    # Set up cell representation
    if cellrep_key == 'X':
        try:
            adata.obsm['nf_cellrep'] = adata.X.toarray()
        except:
            adata.obsm['nf_cellrep'] = adata.X
    else:
        adata.obsm['nf_cellrep'] = adata.obsm[cellrep_key]
    
    # Scale spatial coordinates
    pos = adata.obsm['spatial']
    knn = NearestNeighbors(n_neighbors=neighbor_num)
    knn.fit(pos)
    dists = knn.kneighbors(pos)[0]
    ref_dist = dists[:, -1].mean()
    adata.obsm['spatial'] = adata.obsm['spatial'] / ref_dist
    
    # Split dataset
    if batch_key is not None:
        batchs_uniq = adata.obs[batch_key].unique()
        train_adata_list = []
        val_adata_list = []
        for batch in batchs_uniq:
            batch_adata = adata[adata.obs[batch_key] == batch]
            val_adata = clip_center_adata(batch_adata, 0.1)
            train_adata = batch_adata[~batch_adata.obs_names.isin(val_adata.obs_names)]
            train_adata_list.append(train_adata)
            val_adata_list.append(val_adata)
            print(f'batch {batch} train size: {train_adata.shape[0]}, val size: {val_adata.shape[0]}')
        train_adata = anndata.concat(train_adata_list)
        val_adata = anndata.concat(val_adata_list)
    else:
        val_adata = clip_center_adata(adata, 0.1)
        train_adata = adata[~adata.obs_names.isin(val_adata.obs_names)]
    
    # Reload modules to ensure latest version
    importlib.reload(nf)
    
    # Create datasets and dataloaders
    ds_train = adata2ds(train_adata, neighbor_num=neighbor_num, batch_key=batch_key)
    ds_val = adata2ds(val_adata, neighbor_num=neighbor_num, batch_key=batch_key)
    train_loader = DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    # Setup trainer
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'{log_dir_base}/model_{current_time}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f'{log_dir}/ckpt', exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(dirpath=f'{log_dir}/ckpt')
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name=log_dir)
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=ngpu, 
        accelerator="gpu", 
        callbacks=[EarlyStopping(monitor="val_loss", patience=20), checkpoint_callback], 
        reload_dataloaders_every_n_epochs=1, 
        strategy='ddp_find_unused_parameters_true', 
        logger=logger
    )
    
    # Set up model
    model_construct_params = {
        'input_dim': adata.obsm['nf_cellrep'].shape[1],
        'latent_dim': latent_dim,
        'train_ds': ds_train,
        'val_ds': ds_val,
        'kld_ld': kld_ld,
        'pent_ld': pent_ld,
        'dist_space': dspace,
        'batch_size': batch_size,
        'batch_correct': batch_correct,
        'num_layers': nlayers,
        'head_num': nheads
    }
    
    model = nf.NicheFormer(**model_construct_params)
    
    # Train
    trainer.fit(model, val_dataloaders=val_loader)
    
    # Load best model
    best_model_params = {
        'input_dim': adata.obsm['nf_cellrep'].shape[1],
        'latent_dim': latent_dim,
        'train_ds': ds_train,
        'val_ds': ds_val,
        'batch_correct': batch_correct,
        'num_layers': nlayers,
        'head_num': nheads
    }
    
    model = nf.NicheFormer.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        **best_model_params
    )
    
    # Save model
    torch.save(model.state_dict(), model_save_path)
    
    return model, model_construct_params



def loading_pre_trained_model(model_path, adata, model_id_dict=None):
    model_id_dict = dict(model_id_dict or {})
    latent_dim = int(model_id_dict.get('ldim', 20))
    cellrep_key = model_id_dict.get('crkey', 'X_pca')
    batch_key = model_id_dict.get('bkey')
    batch_correct = resolve_mievformer_batch_correction(
        adata,
        batch_key=batch_key,
        batch_correct=model_id_dict.get('bcorr', 'auto'),
    )
    nlayers = int(model_id_dict.get('nlayers', 3))
    nheads = int(model_id_dict.get('nheads', 1))
    model_dataset = None
    if batch_correct:
        model_dataset = SimpleNamespace(
            batch_uniq=np.arange(adata.obs[batch_key].astype(str).nunique())
        )
    model = nf.NicheFormer(
        input_dim=adata.obsm[cellrep_key].shape[1],
        latent_dim=latent_dim,
        train_ds=model_dataset,
        val_ds=None,
        batch_correct=batch_correct,
        num_layers=nlayers,
        head_num=nheads,
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def gpu_prod(large_mat, small_mat):
    """
    Perform matrix multiplication using GPU.
    
    Parameters
    ----------
    large_mat : torch.Tensor
        The large matrix to be multiplied.
    small_mat : torch.Tensor
        The small matrix to be multiplied.
    
    Returns
    -------
    torch.Tensor
        The result of the matrix multiplication.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    large_mat = torch.tensor(large_mat).float()
    small_mat = torch.tensor(small_mat).float().to(device)
    dataset = torch.utils.data.TensorDataset(large_mat)
    data_loader = DataLoader(dataset, batch_size=1024, num_workers=0, pin_memory=True)
    res_list = []
    for batch in data_loader:
        batch = batch[0].to(device)
        result = batch @ small_mat
        res_list.append(result.cpu())
    result = torch.cat(res_list, dim=0)
    return result

def estimate_population_density(adata, group, cluster_key, max_cell_num=1000):
    """
    Estimate the population density of cells in a given group.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    group : str
        The name of the group to estimate the density for.
    cluster_key : str
        The key for the clustering information in adata.obs.
    
    Returns
    -------
    adata : AnnData
        The updated annotated data matrix with density information. The density is stored in adata.obsm[f'{group}_density'].
    """
    if 'w_z' not in adata.obsm.keys():
        raise ValueError("w_z is not found in adata.obsm. Please run add_wb_ez first.")
    group_idxs = np.where(adata.obs[cluster_key] == group)[0]
    total_group_num = len(group_idxs)
    total_num = adata.shape[0]
    if len(group_idxs) == 0:
        raise ValueError(f"No cells found in group {group}.")
    if len(group_idxs) > max_cell_num:
        group_idxs = np.random.choice(group_idxs, max_cell_num, replace=False)
    ref_idxs = np.random.choice(np.arange(adata.shape[0]), max_cell_num, replace=False)
    group_ws = gpu_prod(adata.obsm['w_e'], adata[group_idxs].obsm['w_z'].T) + adata[group_idxs].obsm['b_z'].flatten()
    ref_ws = gpu_prod(adata.obsm['w_e'], adata[ref_idxs].obsm['w_z'].T) + adata[ref_idxs].obsm['b_z'].flatten()
    raw_density = np.exp(group_ws + np.log(len(ref_idxs)) - logsumexp(ref_ws, axis=1, keepdims=True)).mean(axis=1)
    adata.obs[f'{group}_density'] = np.clip(raw_density * total_group_num / total_num, None, 1)
    return adata



def calculate_niche_density_ratio(adata, ref_niche_num, stratify_key='leiden_e', min_ratio=0.01, ref_adata=None):
    """
    Compute per-cell density ratios over a panel of reference niches.

    For each cell ``i`` and reference niche ``j`` sampled by stratified
    sampling on ``stratify_key``, the log density ratio is

    .. math::
        \\log r_{ij}
        = \\log p(e_j \\mid z_i) - \\log p(e_j)
        = (w_z(z_i)^\\top w_e(e_j) + b_z(z_i))
          - \\log \\sum_{k \\in \\mathrm{ref}}
            \\exp(w_z(z_k)^\\top w_e(e_j) + b_z(z_k)).

    The matrix is then softmax-normalized per cell over reference niches, so
    each row of ``adata.obsm['dist_e']`` is a probability distribution over
    the sampled reference niches that emphasizes niches whose environment
    becomes more likely under the cell's state than under the marginal.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix. Must contain ``w_e``, ``w_z``, ``b_z`` in
        ``obsm`` (produced by :func:`add_wb_ez`).
    ref_niche_num : int
        Number of reference niches to sample.
    stratify_key : str, optional
        ``adata.obs`` column used for stratified sampling. Default ``'leiden_e'``.
    min_ratio : float, optional
        Clusters with frequency below this fraction of the dataset are dropped
        from stratified sampling. Default ``0.01``.
    ref_adata : AnnData, optional
        External reference. If ``None``, reference niches are drawn from ``adata``.

    Returns
    -------
    adata : AnnData
        Updated in-place with
        ``obsm['dist_e']`` — softmax-normalized density ratios of shape
        ``(n_cells, ref_niche_num)`` — and
        ``uns['dist_e']['ref_obs']`` — obs names of the sampled reference niches.
        The ``dist_e`` key name is preserved for backward compatibility with
        existing h5ad artifacts.
    """
    if 'w_z' not in adata.obsm.keys():
        raise ValueError("w_z is not found in adata.obsm. Please run add_wb_ez first.")
    if stratify_key is not None:
        cluster_counts = adata.obs[stratify_key].value_counts()
        stratify_clusters = cluster_counts.index[cluster_counts > (min_ratio * adata.shape[0])]
        nstratify = len(stratify_clusters)
        each_num = ref_niche_num // nstratify
        ref_obs_names = np.concatenate([
            np.random.choice(adata.obs_names[adata.obs[stratify_key] == stratify], each_num, replace=True)
            for stratify in stratify_clusters
        ])
    else:
        ref_obs_names = np.random.choice(adata.obs_names, ref_niche_num, replace=False)
    ref_adata = adata[ref_obs_names]
    ref_w_e = ref_adata.obsm['w_e']
    if ref_adata is None:
        raw_p = log_softmax(gpu_prod(adata.obsm['w_z'], ref_w_e.T) + adata.obsm['b_z'].reshape(-1, 1), axis=0)
    else:
        pre_raw_p = gpu_prod(adata.obsm['w_z'], ref_w_e.T) + adata.obsm['b_z'].reshape(-1, 1)
        pre_raw_p_norm = logsumexp(gpu_prod(ref_adata.obsm['w_z'], ref_w_e.T) + ref_adata.obsm['b_z'].reshape(-1, 1), axis=0)
        raw_p = pre_raw_p - pre_raw_p_norm.reshape(1, -1)
    adata.obsm['dist_e'] = softmax(raw_p, axis=1)
    # adata.obsm['dist_e'] = np.exp(raw_p)
    adata.uns['dist_e'] = {
        'ref_obs': ref_adata.obs_names.tolist()
    }
    return adata



def calculate_niche_communication_strength(adata, niche_cluster_key='leiden_e', ref_niche_num=1000):
    """
    Calculate communication strength between niches based on spatial distributions.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix with dist_e_agg in adata.obsm.
    niche_cluster_key : str, default 'leiden_e'
        The key for niche clustering in adata.obs.
    ref_niche_num : int, default 1000
        Number of reference niches used when computing the niche density ratio.
    
    Returns
    -------
    comm_strength_df : pd.DataFrame
        A matrix of communication strengths between niche clusters.
    """
    if 'dist_e_agg' not in adata.obsm.keys():
        raise ValueError("dist_e_agg is not found in adata.obsm. Please run calculate_niche_cluster_membership first.")
    
    dist_e_agg = adata.obsm['dist_e_agg']
    unique_clusters = adata.obs[niche_cluster_key].unique()
    
    # Calculate communication strength as correlation between distributions
    comm_strength_list = []
    for cluster1 in unique_clusters:
        for cluster2 in unique_clusters:
            if cluster1 in dist_e_agg.columns and cluster2 in dist_e_agg.columns:
                corr = np.corrcoef(dist_e_agg[cluster1], dist_e_agg[cluster2])[0, 1]
                comm_strength_list.append({
                    'source': cluster1,
                    'target': cluster2,
                    'strength': corr
                })
    
    comm_strength_df = pd.DataFrame(comm_strength_list)
    comm_strength_matrix = comm_strength_df.pivot(index='source', columns='target', values='strength')
    
    return comm_strength_matrix


def calculate_niche_specificity_scores(adata, niche_cluster_key='leiden_e', ref_num=1000):
    """
    Calculate niche specificity scores for each cell based on model weights.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix with model weights in adata.obsm.
    niche_cluster_key : str, default 'leiden_e'
        The key for niche clustering in adata.obs.
    ref_num : int, default 1000
        Number of reference cells to use for centroid calculation.
    
    Returns
    -------
    adata : AnnData
        The updated annotated data matrix with niche specificity scores.
    """
    if 'w_e' not in adata.obsm.keys() or 'w_z' not in adata.obsm.keys() or 'b_z' not in adata.obsm.keys():
        raise ValueError("Model weights not found. Please run add_wb_ez first.")
    
    adata = calculate_niche_specificity(adata, niche_cluster_key=niche_cluster_key, ref_num=ref_num)
    
    return adata


def calculate_niche_cluster_membership(adata, group_key='leiden_e'):
    """
    Aggregate per-cell density ratios into a soft membership over niche clusters.

    Averages the columns of ``adata.obsm['dist_e']`` within each value of
    ``adata.obs[group_key]`` (typically ``leiden_e`` niche clusters). The
    resulting ``adata.obsm['dist_e_agg']`` has shape
    ``(n_cells, n_niche_clusters)``; entry ``[i, c]`` is the mean density
    ratio ``p(e|z_i)/p(e)`` evaluated at reference cells in cluster ``c``
    and is used as a soft assignment of cell ``i`` to niche cluster ``c``.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix. Must contain ``obsm['dist_e']`` (see
        :func:`calculate_niche_density_ratio`). If absent, that function is
        invoked with defaults.
    group_key : str, optional
        ``adata.obs`` column with niche cluster labels. Default ``'leiden_e'``.

    Returns
    -------
    adata : AnnData
        Updated with ``obsm['dist_e_agg']``: per-cell niche-cluster membership
        vectors (columns are niche cluster labels). The ``dist_e_agg`` key
        name is preserved for backward compatibility with existing h5ad
        artifacts used by figure scripts.
    """
    if 'dist_e' not in adata.obsm.keys():
        adata = calculate_niche_density_ratio(adata, ref_niche_num=1000, stratify_key=group_key)
        print("Niche density ratio calculated and stored in adata.obsm['dist_e'].")
    dist_e = adata.obsm['dist_e']
    dist_e_agg = pd.DataFrame(dist_e, index=adata.obs_names, columns=adata.uns['dist_e']['ref_obs']).transpose()
    dist_e_agg['group'] = adata[dist_e_agg.index].obs[group_key].values
    dist_e_agg = dist_e_agg.groupby('group').mean()
    # dist_e_agg = dist_e_agg.div(dist_e_agg.mean(axis=0), axis=1)
    adata.obsm['dist_e_agg'] = dist_e_agg.transpose()
    return adata


def basis_clustering(adata, basis, added_key):
    """
    Perform clustering on the basis of a given key.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    basis : str
        The key for the basis information in adata.obsm.
    added_key : str
        The key for the clustering information in adata.obs.
    
    Returns
    -------
    adata : AnnData
        The updated annotated data matrix with clustering information.
    """
    sc.pp.neighbors(adata, use_rep=basis)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added=added_key)
    return adata


def postprocess_nicheformer(
    adata_path: str,
    model_path: str,
    output_niche_rep_path: str,
    output_e_adata_path: str,
    model_dict: dict,
    max_cells: int = 1_000_000,
) -> None:
    """
    NicheFormer で空間的ニッチ表現を計算し、結果を保存するユーティリティ関数。

    Parameters
    ----------
    adata_path : str
        入力 AnnData (.h5ad) ファイルへのパス
    model_path : str
        学習済み NicheFormer モデル (.pt) のパス
    output_niche_rep_path : str
        np.save 形式で保存するニッチ表現 (numpy.ndarray) の出力先
    output_e_adata_path : str
        e (ニッチ表現) を格納した AnnData を書き出す .h5ad パス
    model_dict : dict
        モデルパラメータを含む辞書
    max_cells : int, default 1_000_000
        計算コスト削減のための最大セル数。超える場合はランダムサンプリング
    """

    # ---------- 1. AnnData の読み込み & 前処理 ----------
    adata = sc.read_h5ad(adata_path)
    if adata.shape[0] > max_cells:
        adata = adata[np.random.permutation(adata.shape[0])[:max_cells]]
    adata.obs_names_make_unique()

    # ---------- 2. モデル ID のパラメータ解釈 ----------
    latent_dim   = int(model_dict.get('ldim', 20))
    neighbor_num = int(model_dict.get('nn', 100))
    cellrep_key  = model_dict.get('crkey', 'X_pca')
    batch_correct = model_dict.get('bcorr', 'false').lower() == 'true'
    nlayers      = int(model_dict.get('nlayers', 3))
    nheads       = int(model_dict.get('nheads', 1))
    batch_key    = model_dict.get('bkey', None)

    # ---------- 3. セル表現 (obsm['nf_cellrep']) の準備 ----------
    if cellrep_key == 'X':
        adata.obsm['nf_cellrep'] = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    else:
        adata.obsm['nf_cellrep'] = adata.obsm[cellrep_key]

    # ---------- 4. 空間座標 (obsm['spatial']) の正規化 ----------
    if 'spatial' not in adata.obsm:
        adata.obsm['spatial'] = adata.obs[['centroid_x', 'centroid_y']].values

    pos  = adata.obsm['spatial']
    knn  = NearestNeighbors(n_neighbors=neighbor_num).fit(pos)
    ref_dist = knn.kneighbors(pos, return_distance=True)[0][:, -1].mean()
    adata.obsm['spatial'] = pos / ref_dist

    # ---------- 5. バッチ分割 (必要な場合のみ) ----------
    if batch_key is not None:
        adata = anndata.concat(
            [adata[adata.obs[batch_key] == b].copy() for b in adata.obs[batch_key].cat.categories],
            axis=0,
            uns_merge='unique',
        )

    # ---------- 6. データセット化 & モデル読み込み ----------
    ds = adata2ds(adata, neighbor_num=neighbor_num, batch_key=batch_key)
    model = nf.NicheFormer(
        input_dim=adata.obsm['nf_cellrep'].shape[1],
        latent_dim=latent_dim,
        train_ds=ds,
        val_ds=None,
        batch_correct=batch_correct,
        num_layers=nlayers,
        head_num=nheads,
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # ---------- 7. ニッチ表現の計算 ----------
    e_tensor = utils.output_niche_rep(ds, model)  # shape = [n_cells, latent_dim]
    np.save(output_niche_rep_path, e_tensor.numpy())

    # AnnData に格納
    adata.obsm['e'] = e_tensor.numpy()
    if batch_correct:
        adata.obsm['batch_one_hot'] = torch.stack([ds.batch_one_hots[b] for b in ds.batchs]).numpy()

    # ---------- 8. Web / Batch エンリッチメントなど追加解析 ----------
    adata = add_wb_ez(adata, model, cell_rep_key='nf_cellrep')

    return adata




def sample_cells_by_exression(adata, gene, sample_cells=1000):
    """
    Sample cells based on the expression of a specific gene.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    gene : str
        The gene to sample cells by.
    sample_cells : int, default 1000
        Number of cells to sample.
    
    Returns
    -------
    sampled_adata : AnnData
        The sampled annotated data matrix.
    """
    if gene not in adata.var_names:
        raise ValueError(f"Gene {gene} not found in adata.var_names.")
    if 'log1p' not in adata.uns.keys():
        raise ValueError("adata.var['log1p'] is not found. Please run log1p on adata.X first.")
    gene_expression = np.expm1(adata[:, gene].X.toarray().flatten())
    probs = gene_expression / gene_expression.sum()
    sampled_cells = np.random.choice(adata.obs_names, size=sample_cells, replace=True, p=probs)
    return sampled_cells

def cluster_cells_by_niche_membership(adata, n_clusters=15, use_rep='dist_e_agg', key_added='niche_composition_cluster'):
    from scipy.cluster.hierarchy import linkage, fcluster
    if use_rep not in adata.obsm:
        raise ValueError(f"{use_rep} not found in adata.obsm")
    
    X = adata.obsm[use_rep]
    # Check if X is DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.values
        
    # Ward clustering
    # For large datasets, linkage might be slow. But dist_e_agg is usually for subset or aggregated?
    # The reference script uses it on `adata_sub`.
    # If X is too large, we might need to warn or sample.
    # Assuming X is manageable size as per reference script usage.
    Z = linkage(X, method='ward')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    adata.obs[key_added] = labels.astype(str)
    return adata

def plot_niche_membership_clustermap(adata, cluster_key='niche_composition_cluster', use_rep='dist_e_agg', file_path=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    if use_rep not in adata.obsm:
        raise ValueError(f"{use_rep} not found in adata.obsm")
    if cluster_key not in adata.obs:
        raise ValueError(f"{cluster_key} not found in adata.obs")
        
    X = adata.obsm[use_rep]
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, index=adata.obs_names)
        
    cluster_labels = adata.obs[cluster_key]
    
    # Create row colors
    unique_clusters = np.unique(cluster_labels)
    # Use tab20 or similar
    if len(unique_clusters) <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
        
    cluster2color = {cl: col for cl, col in zip(unique_clusters, colors)}
    row_colors = cluster_labels.map(cluster2color)
    
    g = sns.clustermap(
        X,
        row_cluster=True,
        col_cluster=False,
        cmap='viridis',
        yticklabels=False,
        figsize=(12, 10),
        method='ward',
        row_colors=row_colors,
        cbar_kws={'label': 'Estimated Density'}
    )
    
    g.ax_heatmap.set_xlabel('Niche Cluster')
    g.ax_heatmap.set_ylabel('Cells')
    
    # Legend
    handles = [Patch(facecolor=cluster2color[cl], label=f'Cluster {cl}') for cl in unique_clusters]
    plt.legend(handles=handles, title='Spatial Cluster', bbox_to_anchor=(1.15, 1), bbox_transform=plt.gcf().transFigure)
    
    if file_path:
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
    return g

def _require_reference_probability_ca_inputs(
    adata,
    w_e_key='w_e',
    w_z_key='w_z',
    b_z_key='b_z',
):
    """Validate the model-derived arrays required by reference-probability CA."""
    missing = [key for key in [w_e_key, w_z_key, b_z_key] if key not in adata.obsm]
    if missing:
        raise ValueError(
            f"Missing Mievformer weight arrays in adata.obsm: {missing}. "
            "Run add_wb_ez first; no non-Mievformer fallback is used."
        )

    w_e_shape = tuple(adata.obsm[w_e_key].shape)
    w_z_shape = tuple(adata.obsm[w_z_key].shape)
    b_z_shape = tuple(adata.obsm[b_z_key].shape)
    if len(w_e_shape) != 2 or len(w_z_shape) != 2:
        raise ValueError('w_e and w_z must be two-dimensional')
    if w_e_shape[0] != adata.n_obs or w_z_shape[0] != adata.n_obs:
        raise ValueError('w_e and w_z must contain one row per observation')
    if w_e_shape[1] != w_z_shape[1]:
        raise ValueError('w_e and w_z must have the same latent dimension')
    if int(np.prod(b_z_shape)) != adata.n_obs:
        raise ValueError('b_z must contain one value per observation')


def _reference_probability_ca_device(device):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resolved = torch.device(device)
    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested for reference-probability CA, but it is unavailable')
    return resolved


def _reference_probability_batches(
    query_w_e,
    reference_w_z,
    reference_b_z,
    batch_size,
    device,
):
    """Yield stable common-reference softmax probabilities in query batches."""
    if int(batch_size) < 1:
        raise ValueError('batch_size must be at least 1')
    reference_w_z = np.asarray(reference_w_z, dtype=np.float32)
    reference_b_z = np.asarray(reference_b_z, dtype=np.float32).reshape(-1)
    if reference_w_z.ndim != 2:
        raise ValueError('reference_w_z must be two-dimensional')
    if reference_w_z.shape[0] != reference_b_z.shape[0]:
        raise ValueError('reference_w_z and reference_b_z have incompatible shapes')
    if len(query_w_e.shape) != 2 or query_w_e.shape[1] != reference_w_z.shape[1]:
        raise ValueError('query_w_e and reference_w_z have incompatible shapes')
    if not np.isfinite(reference_w_z).all():
        raise ValueError('reference_w_z must contain only finite values')
    if not np.isfinite(reference_b_z).all():
        raise ValueError('reference_b_z must contain only finite values')

    if device.type == 'cuda':
        reference_w_z_device = torch.as_tensor(
            reference_w_z,
            dtype=torch.float32,
            device=device,
        )
        reference_b_z_device = torch.as_tensor(
            reference_b_z,
            dtype=torch.float32,
            device=device,
        )
    else:
        reference_w_z_t = np.ascontiguousarray(reference_w_z.T)

    with torch.no_grad():
        for start in range(0, int(query_w_e.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(query_w_e.shape[0]))
            query_batch = np.asarray(query_w_e[start:stop], dtype=np.float32)
            if not np.isfinite(query_batch).all():
                raise ValueError(f'query_w_e contains non-finite values in rows {start}:{stop}')
            if device.type == 'cuda':
                query_device = torch.as_tensor(
                    query_batch,
                    dtype=torch.float32,
                    device=device,
                )
                logits = query_device @ reference_w_z_device.T
                logits = logits + reference_b_z_device
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                del query_device, logits
            else:
                logits = query_batch @ reference_w_z_t
                logits += reference_b_z[None, :]
                logits -= logits.max(axis=1, keepdims=True)
                np.exp(logits, out=logits)
                logits /= logits.sum(axis=1, keepdims=True)
                probabilities = logits
            probabilities = np.asarray(probabilities, dtype=np.float32)
            if not np.isfinite(probabilities).all():
                raise ValueError('Reference probabilities contain non-finite values')
            row_sum_error = float(
                np.max(np.abs(probabilities.sum(axis=1, dtype=np.float64) - 1.0))
            )
            if row_sum_error > 1.0e-4:
                raise RuntimeError(
                    f'Reference probability row sums are invalid: max error={row_sum_error}'
                )
            yield start, stop, probabilities


def calculate_reference_probabilities(
    query_w_e,
    reference_w_z,
    reference_b_z,
    batch_size=2048,
    device=None,
):
    """Calculate ``p(reference | query)`` under one common reference set.

    This low-level helper materializes the complete probability matrix and is
    therefore intended for fitting subsets or small datasets. Use
    :func:`transform_reference_probability_ca` for full-dataset projection,
    which never materializes the full ``n_cells x n_references`` matrix.
    """
    resolved_device = _reference_probability_ca_device(device)
    probabilities = np.empty(
        (int(query_w_e.shape[0]), int(np.asarray(reference_b_z).size)),
        dtype=np.float32,
    )
    for start, stop, batch in _reference_probability_batches(
        query_w_e,
        reference_w_z,
        reference_b_z,
        batch_size,
        resolved_device,
    ):
        probabilities[start:stop] = batch
    return probabilities


def _fit_reference_probability_ca_basis(probabilities, max_components, device=None):
    """Fit an exact covariance CA basis to a common-reference probability table."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 2:
        raise ValueError('probabilities must be two-dimensional')
    if not np.isfinite(probabilities).all():
        raise ValueError('probabilities must contain only finite values')
    n_fit, n_references = probabilities.shape
    if n_fit < 2:
        raise ValueError('Reference-probability CA requires at least two fitting cells')
    max_rank = min(n_fit - 1, n_references - 1)
    if not 1 <= int(max_components) <= max_rank:
        raise ValueError(
            f'max_components must be between 1 and {max_rank}; got {max_components}'
        )

    column_mass = probabilities.mean(axis=0, dtype=np.float64)
    if not np.isfinite(column_mass).all() or np.any(column_mass <= 0):
        invalid = int(np.sum(~np.isfinite(column_mass) | (column_mass <= 0)))
        raise ValueError(
            f'CA requires finite positive reference column masses; found {invalid} invalid'
        )
    column_mass = column_mass.astype(np.float32)
    if not np.isfinite(column_mass).all() or np.any(column_mass <= 0):
        invalid = int(np.sum(~np.isfinite(column_mass) | (column_mass <= 0)))
        raise ValueError(
            'CA requires finite positive float32 reference column masses; '
            f'found {invalid} invalid'
        )
    inverse_sqrt_mass = np.reciprocal(np.sqrt(column_mass)).astype(np.float32)
    if not np.isfinite(inverse_sqrt_mass).all():
        raise ValueError('CA inverse square-root column masses must be finite')
    resolved_device = _reference_probability_ca_device(device)
    if resolved_device.type == 'cuda':
        standardized = torch.as_tensor(
            probabilities,
            dtype=torch.float32,
            device=resolved_device,
        )
        column_mass_device = torch.as_tensor(
            column_mass,
            dtype=torch.float32,
            device=resolved_device,
        )
        standardized = standardized - column_mass_device[None, :]
        standardized = standardized * torch.rsqrt(column_mass_device)[None, :]
        covariance = standardized.T @ standardized
        covariance /= float(n_fit - 1)
        covariance = (covariance + covariance.T) * 0.5
        if not bool(torch.isfinite(covariance).all().item()):
            raise ValueError('CA covariance contains non-finite values')
        total_variance = float(torch.trace(covariance).item())
        if not np.isfinite(total_variance) or total_variance <= 0:
            raise ValueError(f'CA covariance has invalid total variance: {total_variance}')
        all_eigenvalues, all_components = torch.linalg.eigh(covariance)
        if not bool(torch.isfinite(all_eigenvalues).all().item()):
            raise ValueError('CA eigensolver returned non-finite eigenvalues')
        if not bool(torch.isfinite(all_components).all().item()):
            raise ValueError('CA eigensolver returned non-finite components')
        eigenvalues = (
            all_eigenvalues[-int(max_components):]
            .flip(0)
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        components = (
            all_components[:, -int(max_components):]
            .flip(1)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        del standardized, column_mass_device, covariance
        del all_eigenvalues, all_components
        torch.cuda.empty_cache()
    else:
        standardized = probabilities - column_mass[None, :]
        standardized *= inverse_sqrt_mass[None, :]
        covariance = standardized.T @ standardized
        covariance /= float(n_fit - 1)
        covariance = (covariance + covariance.T) * np.float32(0.5)
        if not np.isfinite(covariance).all():
            raise ValueError('CA covariance contains non-finite values')
        total_variance = float(np.trace(covariance, dtype=np.float64))
        if not np.isfinite(total_variance) or total_variance <= 0:
            raise ValueError(f'CA covariance has invalid total variance: {total_variance}')
        first_index = n_references - int(max_components)
        eigenvalues, components = linalg.eigh(
            covariance,
            subset_by_index=[first_index, n_references - 1],
            check_finite=False,
            driver='evr',
        )
        eigenvalues = np.asarray(eigenvalues[::-1], dtype=np.float64)
        components = np.asarray(components[:, ::-1], dtype=np.float32)
    if not np.isfinite(eigenvalues).all():
        raise ValueError('The requested CA spectrum contains non-finite eigenvalues')
    if not np.isfinite(components).all():
        raise ValueError('The requested CA components contain non-finite values')
    # CPU and GPU metadata both refer only to the selected requested spectrum.
    negative_count = int(np.sum(eigenvalues < 0))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    if np.any(eigenvalues <= 0):
        raise ValueError('The requested CA spectrum contains non-positive eigenvalues')

    # Fix the otherwise arbitrary eigenvector signs for reproducible artifacts.
    pivot_rows = np.argmax(np.abs(components), axis=0)
    signs = np.sign(components[pivot_rows, np.arange(components.shape[1])])
    signs[signs == 0] = 1
    components *= signs[None, :]
    if not np.isfinite(components).all():
        raise ValueError('Canonicalized CA components contain non-finite values')
    explained_variance_ratio = eigenvalues / total_variance
    if not np.isfinite(explained_variance_ratio).all():
        raise ValueError('CA explained-variance ratios contain non-finite values')
    return {
        'column_mass': column_mass,
        'components': components,
        'explained_variance': eigenvalues,
        'explained_variance_ratio': explained_variance_ratio,
        'total_variance': total_variance,
        'negative_eigenvalue_count': negative_count,
    }


def _reference_probability_ca_log_chord(values, candidate_dimensions):
    values = np.asarray(values, dtype=np.float64)
    x = np.arange(1, values.size + 1, dtype=np.float64)
    log_values = np.log(values)
    denominator = float(log_values[0] - log_values[-1])
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('CA spectrum must decrease for log-chord selection')
    x_norm = (x - x[0]) / (x[-1] - x[0])
    y_norm = (log_values - log_values[-1]) / denominator
    distance = np.abs(y_norm - (1.0 - x_norm))
    candidates = np.asarray(candidate_dimensions, dtype=int)
    return int(candidates[np.argmax(distance[candidates - 1])])


def _reference_probability_ca_l_method(values, candidate_dimensions):
    values = np.asarray(values, dtype=np.float64)
    x = np.arange(1, values.size + 1, dtype=np.float64)
    y = np.log(values)
    rows = []
    for dimension in np.asarray(candidate_dimensions, dtype=int):
        error = 0.0
        for xx, yy in [
            (x[:dimension], y[:dimension]),
            (x[dimension - 1:], y[dimension - 1:]),
        ]:
            coefficients = np.polyfit(xx, yy, 1)
            error += float(np.sum(np.square(yy - np.polyval(coefficients, xx))))
        rows.append((error, int(dimension)))
    if not rows:
        raise ValueError('No valid CA L-method breakpoint candidates')
    return min(rows)[1]


_REFERENCE_PROBABILITY_CA_SNAP_TIE_ATOL = 1.0e-12


def _snap_reference_probability_ca_dimension(
    value,
    standard_dimensions,
    tie_atol=_REFERENCE_PROBABILITY_CA_SNAP_TIE_ATOL,
):
    """Snap to the standard grid, resolving exact numerical ties downward."""
    value = float(value)
    standard_dimensions = np.asarray(standard_dimensions, dtype=int)
    if not np.isfinite(value):
        raise ValueError('CA selector dimension must be finite before grid snapping')
    if standard_dimensions.ndim != 1 or standard_dimensions.size == 0:
        raise ValueError('At least one standard CA dimension is required for grid snapping')
    if not np.isfinite(tie_atol) or float(tie_atol) < 0:
        raise ValueError('CA grid-snap tie tolerance must be finite and non-negative')
    distances = np.abs(standard_dimensions.astype(np.float64) - value)
    minimum_distance = float(np.min(distances))
    tied_dimensions = standard_dimensions[
        np.abs(distances - minimum_distance) <= float(tie_atol)
    ]
    return int(np.min(tied_dimensions))


def select_reference_probability_ca_dimension(
    reference_spectra,
    candidate_dimensions=(5, 10, 20, 40),
    expected_n_reference_spectra=None,
):
    """Select CA dimensionality from the mean relative eigengap without GT.

    Spectra from every planned reference-cell sample are averaged. The largest
    relative eigengap after components 2 through rank - 1 is snapped to the
    nearest predeclared standard dimension, with exact grid ties resolved to
    the smaller dimension.
    """
    spectra = np.asarray(reference_spectra, dtype=np.float64)
    if spectra.ndim == 1:
        spectra = spectra[None, :]
    if spectra.ndim != 2 or spectra.shape[0] < 1 or spectra.shape[1] < 3:
        raise ValueError('reference_spectra must have shape (n_repeats, >=3)')
    if expected_n_reference_spectra is None:
        expected_n_reference_spectra = int(spectra.shape[0])
    expected_n_reference_spectra = int(expected_n_reference_spectra)
    if expected_n_reference_spectra < 1:
        raise ValueError('expected_n_reference_spectra must be at least 1')
    if spectra.shape[0] != expected_n_reference_spectra:
        raise ValueError(
            'All planned reference spectra are required: '
            f'expected {expected_n_reference_spectra}, observed {spectra.shape[0]}'
        )
    if not np.isfinite(spectra).all() or np.any(spectra <= 0):
        raise ValueError('reference_spectra must contain finite positive eigenvalues')

    mean_spectrum = spectra.mean(axis=0)
    if not np.isfinite(mean_spectrum).all() or np.any(mean_spectrum <= 0):
        raise ValueError('mean reference spectrum must contain finite positive eigenvalues')
    standard_dimensions = np.asarray(
        sorted(set(int(value) for value in candidate_dimensions)),
        dtype=int,
    )
    standard_dimensions = standard_dimensions[
        (standard_dimensions >= 1) & (standard_dimensions <= mean_spectrum.size)
    ]
    if standard_dimensions.size == 0:
        raise ValueError('No candidate_dimensions are available in the fitted spectrum')
    breakpoint_candidates = np.arange(2, mean_spectrum.size, dtype=int)
    relative_gaps = mean_spectrum[:-1] / mean_spectrum[1:] - 1.0
    if not np.isfinite(relative_gaps).all():
        raise ValueError('Mean reference spectrum produced non-finite relative eigengaps')
    eigengap_dimension = int(
        breakpoint_candidates[
            np.argmax(relative_gaps[breakpoint_candidates - 1])
        ]
    )
    selector_dimensions = {'mean_relative_eigengap': eigengap_dimension}
    selected = _snap_reference_probability_ca_dimension(
        eigengap_dimension,
        standard_dimensions,
    )
    selector_grid_dimensions = {
        name: _snap_reference_probability_ca_dimension(
            dimension,
            standard_dimensions,
        )
        for name, dimension in selector_dimensions.items()
    }
    agreeing_selectors = [
        name
        for name, dimension in selector_grid_dimensions.items()
        if dimension == selected
    ]
    minimum_selector_grid_agreement = 1
    selector_grid_agreement_count = len(agreeing_selectors)
    selection_accepted = True
    diagnostics = {
        'selection_rule': 'mean_relative_eigengap_snapped_to_standard_grid',
        'selection_formula': (
            'selected_grid_dimension = snap_to_standard_grid('
            'argmax_d(mean_lambda_d / mean_lambda_(d+1) - 1), '
            'd in 2..rank-1)'
        ),
        'acceptance_formula': (
            'accept = all_planned_reference_spectra_present_finite_positive'
        ),
        'selected_n_components': selected,
        'candidate_dimensions': standard_dimensions.tolist(),
        'selector_dimensions': selector_dimensions,
        'selector_grid_dimensions': selector_grid_dimensions,
        'raw_selected_n_components': eigengap_dimension,
        'minimum_selector_grid_agreement': minimum_selector_grid_agreement,
        'selector_grid_agreement_count': selector_grid_agreement_count,
        'selector_grid_agreement_fraction': (
            selector_grid_agreement_count / float(len(selector_dimensions))
        ),
        'agreeing_selectors': agreeing_selectors,
        'selection_accepted': selection_accepted,
        'reference_spectrum_positive_threshold_exclusive': 0.0,
        'expected_n_reference_spectra': expected_n_reference_spectra,
        'n_reference_spectra': int(spectra.shape[0]),
        'all_planned_reference_spectra_present': True,
        'all_reference_spectra_finite': True,
        'all_reference_spectra_positive': True,
        'snap_tie_absolute_tolerance': (
            _REFERENCE_PROBABILITY_CA_SNAP_TIE_ATOL
        ),
        'snap_tie_rule': (
            'choose_smaller_dimension_when_grid_distances_tie_within_absolute_tolerance'
        ),
        'mean_explained_variance': mean_spectrum.tolist(),
        # There is no eigengap after the final component.  Store only the
        # mathematically defined finite values so serialized diagnostics never
        # require a NaN sentinel.
        'mean_relative_eigengap_after_component': relative_gaps.tolist(),
        'ground_truth_used': False,
        'cluster_count_used': False,
    }
    return selected, diagnostics


def fit_reference_probability_ca(
    adata,
    reference_num=1000,
    reference_seed=0,
    n_components='auto',
    dimension_reference_seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    candidate_dimensions=(5, 10, 20, 40),
    max_spectrum_components=40,
    max_fit_cells=50000,
    fit_seed=0,
    w_e_key='w_e',
    w_z_key='w_z',
    b_z_key='b_z',
    query_batch_size=2048,
    device=None,
):
    """Fit a reusable correspondence-analysis model to reference probabilities.

    Reference cells are sampled from ``adata`` and define one shared softmax
    denominator for every query cell. CA is fitted on a deterministic query
    subset when ``n_obs > max_fit_cells``. The returned model freezes the
    reference weights, column masses, and CA loadings needed to project new
    cells into exactly the same feature space.
    """
    _require_reference_probability_ca_inputs(
        adata,
        w_e_key=w_e_key,
        w_z_key=w_z_key,
        b_z_key=b_z_key,
    )
    reference_num = int(reference_num)
    reference_seed = int(reference_seed)
    if reference_num < 2 or reference_num > adata.n_obs:
        raise ValueError(
            f'reference_num must be between 2 and n_obs={adata.n_obs}; got {reference_num}'
        )
    if max_fit_cells is None:
        n_fit = int(adata.n_obs)
    else:
        n_fit = min(int(max_fit_cells), int(adata.n_obs))
    if n_fit < 3:
        raise ValueError('At least three fitting cells are required')
    if n_fit == adata.n_obs:
        fit_indices = np.arange(adata.n_obs, dtype=np.int64)
    else:
        fit_rng = np.random.default_rng(int(fit_seed))
        fit_indices = np.sort(
            fit_rng.choice(adata.n_obs, size=n_fit, replace=False)
        ).astype(np.int64)

    w_e = adata.obsm[w_e_key]
    w_z = adata.obsm[w_z_key]
    b_z = adata.obsm[b_z_key]
    fit_w_e = np.asarray(w_e[fit_indices], dtype=np.float32)
    resolved_device = _reference_probability_ca_device(device)
    max_rank = min(n_fit - 1, reference_num - 1)

    auto_dimension = isinstance(n_components, str)
    if auto_dimension and str(n_components).lower() != 'auto':
        raise ValueError("n_components must be an integer or 'auto'")
    if auto_dimension:
        max_components = min(int(max_spectrum_components), max_rank)
        if max_components < 3:
            raise ValueError('Automatic CA dimension selection requires at least 3 components')
        reference_seeds = []
        for seed in dimension_reference_seeds:
            seed = int(seed)
            if seed not in reference_seeds:
                reference_seeds.append(seed)
        if not reference_seeds:
            raise ValueError('dimension_reference_seeds must not be empty')
    else:
        selected_components = int(n_components)
        if not 1 <= selected_components <= max_rank:
            raise ValueError(
                f'n_components must be between 1 and {max_rank}; got {selected_components}'
            )
        max_components = selected_components
        reference_seeds = [reference_seed]

    spectra = []
    final_basis = None
    final_reference_indices = None
    final_reference_w_z = None
    final_reference_b_z = None
    for seed in reference_seeds:
        reference_rng = np.random.default_rng(seed)
        reference_indices = np.sort(
            reference_rng.choice(adata.n_obs, size=reference_num, replace=False)
        ).astype(np.int64)
        reference_w_z = np.asarray(w_z[reference_indices], dtype=np.float32)
        reference_b_z = np.asarray(b_z[reference_indices], dtype=np.float32).reshape(-1)
        probabilities = calculate_reference_probabilities(
            fit_w_e,
            reference_w_z,
            reference_b_z,
            batch_size=query_batch_size,
            device=resolved_device,
        )
        basis = _fit_reference_probability_ca_basis(
            probabilities,
            max_components=max_components,
            device=resolved_device,
        )
        spectra.append(basis['explained_variance'])
        if seed == reference_seed:
            final_basis = basis
            final_reference_indices = reference_indices
            final_reference_w_z = reference_w_z
            final_reference_b_z = reference_b_z

    if auto_dimension:
        if len(spectra) != len(reference_seeds):
            raise RuntimeError(
                'Every unique planned dimension-reference seed must produce a spectrum: '
                f'expected {len(reference_seeds)}, observed {len(spectra)}'
            )
        selected_components, selection = select_reference_probability_ca_dimension(
            np.vstack(spectra),
            candidate_dimensions=candidate_dimensions,
            expected_n_reference_spectra=len(reference_seeds),
        )
        selection['planned_reference_seeds'] = list(reference_seeds)
    else:
        selection = {
            'selection_rule': 'fixed_n_components',
            'selected_n_components': selected_components,
            'candidate_dimensions': [selected_components],
            'selector_dimensions': {},
            'selector_consensus': float(selected_components),
            'n_reference_spectra': 1,
            'mean_explained_variance': spectra[0].tolist(),
            'mean_relative_eigengap_after_component': [],
            'ground_truth_used': False,
            'cluster_count_used': False,
        }

    if final_basis is None:
        reference_rng = np.random.default_rng(reference_seed)
        final_reference_indices = np.sort(
            reference_rng.choice(adata.n_obs, size=reference_num, replace=False)
        ).astype(np.int64)
        final_reference_w_z = np.asarray(
            w_z[final_reference_indices],
            dtype=np.float32,
        )
        final_reference_b_z = np.asarray(
            b_z[final_reference_indices],
            dtype=np.float32,
        ).reshape(-1)
        probabilities = calculate_reference_probabilities(
            fit_w_e,
            final_reference_w_z,
            final_reference_b_z,
            batch_size=query_batch_size,
            device=resolved_device,
        )
        final_basis = _fit_reference_probability_ca_basis(
            probabilities,
            max_components=max(max_components, selected_components),
            device=resolved_device,
        )

    reference_obs_names = np.asarray(
        adata.obs_names.astype(str)[final_reference_indices],
        dtype=str,
    )
    return {
        'method': 'mievformer_reference_probability_correspondence_analysis',
        'version': 2,
        'w_e_key': str(w_e_key),
        'w_z_key': str(w_z_key),
        'b_z_key': str(b_z_key),
        'reference_num': reference_num,
        'reference_seed': reference_seed,
        'reference_indices': final_reference_indices,
        'reference_obs_names': reference_obs_names,
        'reference_w_z': final_reference_w_z,
        'reference_b_z': final_reference_b_z,
        'column_mass': final_basis['column_mass'],
        'components': final_basis['components'][:, :selected_components],
        'selected_n_components': int(selected_components),
        'explained_variance': final_basis['explained_variance'],
        'explained_variance_ratio': final_basis['explained_variance_ratio'],
        'total_variance': float(final_basis['total_variance']),
        'negative_eigenvalue_count': int(final_basis['negative_eigenvalue_count']),
        'fit_indices': fit_indices,
        'n_fit_cells': int(n_fit),
        'fit_seed': int(fit_seed),
        'dimension_reference_seeds': np.asarray(reference_seeds, dtype=np.int64),
        'reference_spectra': np.vstack(spectra),
        'dimension_selection': selection,
        'query_batch_size': int(query_batch_size),
        'fit_device': str(resolved_device),
    }


def validate_reference_probability_ca_dimension_artifact(
    scores,
    ca_model,
    expected_n_components,
):
    """Validate that saved scores and a CA model implement the expected d."""
    expected = int(expected_n_components)
    score_array = np.asarray(scores)
    components = np.asarray(ca_model.get('components'))
    selected = int(ca_model.get('selected_n_components', -1))
    if expected < 1:
        raise ValueError('expected_n_components must be positive')
    if score_array.ndim != 2 or score_array.shape[1] != expected:
        raise ValueError(
            'CA score dimension mismatch: '
            f'expected {expected}, observed {score_array.shape}'
        )
    if components.ndim != 2 or components.shape[1] != expected:
        raise ValueError(
            'CA basis dimension mismatch: '
            f'expected {expected}, observed {components.shape}'
        )
    if selected != expected:
        raise ValueError(
            'CA model selected_n_components mismatch: '
            f'expected {expected}, observed {selected}'
        )
    if not np.isfinite(score_array).all() or not np.isfinite(components).all():
        raise ValueError('CA dimension artifact contains non-finite values')
    return True


def derive_reference_probability_ca_dimension(
    scores,
    ca_model,
    n_components,
    dimension_selection=None,
):
    """Derive a smaller nested CA representation without refitting or transforming."""
    selected = int(n_components)
    source_scores = np.asarray(scores)
    source_components = np.asarray(ca_model.get('components'))
    source_dimension = int(ca_model.get('selected_n_components', -1))
    if source_dimension < 1 or selected < 1 or selected > source_dimension:
        raise ValueError(
            'Derived CA dimension must be in [1, source selected dimension]: '
            f'requested {selected}, source {source_dimension}'
        )
    if source_scores.ndim != 2 or source_scores.shape[1] != source_dimension:
        raise ValueError('Saved CA scores do not match the source selected dimension')
    if source_components.ndim != 2 or source_components.shape[1] != source_dimension:
        raise ValueError('Saved CA basis does not match the source selected dimension')
    derived_scores = np.asarray(source_scores[:, :selected], dtype=source_scores.dtype)
    derived_model = dict(ca_model)
    derived_model['components'] = np.asarray(
        source_components[:, :selected], dtype=source_components.dtype
    )
    derived_model['selected_n_components'] = selected
    if dimension_selection is not None:
        selection = dict(dimension_selection)
        if int(selection.get('selected_n_components', selected)) != selected:
            raise ValueError('dimension_selection does not match derived dimension')
        derived_model['dimension_selection'] = selection
    elif isinstance(derived_model.get('dimension_selection'), dict):
        selection = dict(derived_model['dimension_selection'])
        selection['selected_n_components'] = selected
        selection['derived_from_selected_n_components'] = source_dimension
        derived_model['dimension_selection'] = selection
    validate_reference_probability_ca_dimension_artifact(
        derived_scores,
        derived_model,
        selected,
    )
    return derived_scores, derived_model


def transform_reference_probability_ca(
    adata,
    ca_model,
    w_e_key=None,
    query_batch_size=None,
    device=None,
):
    """Project cells with a fitted reference-probability CA model."""
    if w_e_key is None:
        w_e_key = str(ca_model.get('w_e_key', 'w_e'))
    if w_e_key not in adata.obsm:
        raise ValueError(
            f"obsm['{w_e_key}'] is required for CA projection. "
            "Run add_wb_ez first; no fallback is used."
        )
    required = ['reference_w_z', 'reference_b_z', 'column_mass', 'components']
    missing = [key for key in required if key not in ca_model]
    if missing:
        raise ValueError(f'CA model is missing required fields: {missing}')

    query_w_e = adata.obsm[w_e_key]
    reference_w_z = np.asarray(ca_model['reference_w_z'], dtype=np.float32)
    reference_b_z = np.asarray(ca_model['reference_b_z'], dtype=np.float32).reshape(-1)
    column_mass = np.asarray(ca_model['column_mass'], dtype=np.float32).reshape(-1)
    components = np.asarray(ca_model['components'], dtype=np.float32)
    if reference_w_z.ndim != 2:
        raise ValueError('CA model reference_w_z must be two-dimensional')
    if components.ndim != 2 or components.shape[1] < 1:
        raise ValueError('CA model components must be a non-empty two-dimensional array')
    if (
        reference_w_z.shape[0] != reference_b_z.size
        or reference_w_z.shape[0] != column_mass.size
        or components.shape[0] != column_mass.size
    ):
        raise ValueError('CA model reference arrays have incompatible shapes')
    for name, values in [
        ('reference_w_z', reference_w_z),
        ('reference_b_z', reference_b_z),
        ('column_mass', column_mass),
        ('components', components),
    ]:
        if not np.isfinite(values).all():
            raise ValueError(f'CA model {name} must contain only finite values')
    if np.any(column_mass <= 0) or not np.isfinite(column_mass).all():
        raise ValueError('CA model column masses must be finite and positive')
    if query_batch_size is None:
        query_batch_size = int(ca_model.get('query_batch_size', 2048))
    resolved_device = _reference_probability_ca_device(device)
    scores = np.empty((adata.n_obs, components.shape[1]), dtype=np.float32)
    inverse_sqrt_mass = np.reciprocal(np.sqrt(column_mass)).astype(np.float32)
    if not np.isfinite(inverse_sqrt_mass).all():
        raise ValueError('CA model inverse square-root column masses must be finite')
    for start, stop, probabilities in _reference_probability_batches(
        query_w_e,
        reference_w_z,
        reference_b_z,
        query_batch_size,
        resolved_device,
    ):
        probabilities -= column_mass[None, :]
        probabilities *= inverse_sqrt_mass[None, :]
        scores[start:stop] = probabilities @ components
    if not np.isfinite(scores).all():
        raise ValueError('Projected CA scores contain non-finite values')
    return scores


def add_reference_probability_ca_features(
    adata,
    model=None,
    cell_rep_key='X_pca',
    feature_key='reference_probability_ca',
    copy=False,
    **fit_kwargs,
):
    """Fit and add scalable Mievformer reference-probability CA features.

    If ``w_e``, ``w_z``, or ``b_z`` is absent, an explicit Mievformer model is
    required and the exact weights are calculated with :func:`add_wb_ez`.
    """
    if copy:
        adata = adata.copy()
    weight_keys = [
        fit_kwargs.get('w_e_key', 'w_e'),
        fit_kwargs.get('w_z_key', 'w_z'),
        fit_kwargs.get('b_z_key', 'b_z'),
    ]
    if not all(key in adata.obsm for key in weight_keys):
        if model is None:
            raise ValueError(
                f'Missing Mievformer weight arrays {weight_keys}; provide model or run add_wb_ez'
            )
        if weight_keys != ['w_e', 'w_z', 'b_z']:
            raise ValueError('Custom weight keys cannot be generated by add_wb_ez')
        adata = add_wb_ez(adata, model, cell_rep_key=cell_rep_key)
    ca_model = fit_reference_probability_ca(adata, **fit_kwargs)
    adata.obsm[feature_key] = transform_reference_probability_ca(
        adata,
        ca_model,
        w_e_key=fit_kwargs.get('w_e_key', 'w_e'),
        query_batch_size=fit_kwargs.get('query_batch_size', None),
        device=fit_kwargs.get('device', None),
    )
    adata.uns[feature_key] = ca_model
    return adata, ca_model


def project_reference_probability_ca_features(
    adata,
    ca_model,
    feature_key='reference_probability_ca',
    w_e_key=None,
    query_batch_size=None,
    device=None,
    copy=False,
):
    """Add CA scores to new cells using a frozen fitted CA model."""
    if copy:
        adata = adata.copy()
    adata.obsm[feature_key] = transform_reference_probability_ca(
        adata,
        ca_model,
        w_e_key=w_e_key,
        query_batch_size=query_batch_size,
        device=device,
    )
    adata.uns[feature_key] = ca_model
    return adata


def _mievformer_sample_context(adata, sample_key=None):
    """Return the effective sample key and its persisted/observed order."""
    contract = adata.uns.get('mievformer_batch_contract', {})
    if sample_key == 'same':
        sample_key = None
    if sample_key is None and isinstance(contract, Mapping):
        contract_key = contract.get('batch_key')
        if contract_key in adata.obs:
            sample_key = str(contract_key)
    if sample_key is None:
        return None, ()
    sample_key = str(sample_key)
    if sample_key not in adata.obs:
        raise KeyError(f"obs[{sample_key!r}] not found")
    values = pd.Series(adata.obs[sample_key], index=adata.obs_names)
    if values.isna().any():
        raise ValueError(f"obs[{sample_key!r}] contains missing sample labels")
    observed = values.astype(str).to_numpy()
    observed_set = set(observed)
    if not observed_set:
        raise ValueError(f"obs[{sample_key!r}] does not contain any samples")

    contract_order = []
    if isinstance(contract, Mapping) and contract.get('batch_key') == sample_key:
        contract_order = [str(value) for value in contract.get('batch_order', [])]
    if contract_order:
        if len(contract_order) != len(set(contract_order)):
            raise ValueError('Persisted Mievformer batch order contains duplicates')
        if set(contract_order) != observed_set:
            raise ValueError(
                'Persisted Mievformer batch order does not match observed samples'
            )
        order = tuple(contract_order)
    else:
        order = tuple(str(value) for value in pd.unique(observed))
    return sample_key, order


def resolve_mievformer_ca_strategy(adata, sample_key=None, strategy='auto'):
    """Resolve the standard CA strategy from the number of observed slices.

    ``auto`` selects ordinary Mievformer reference-probability CA for a single
    slice and sample-conditional CA for two or more slices.  Explicit strategy
    selection remains available for controlled method-comparison benchmarks.
    """
    effective_sample_key, sample_order = _mievformer_sample_context(
        adata,
        sample_key=sample_key,
    )
    n_samples = max(1, len(sample_order))
    aliases = {
        'auto': 'auto',
        'single': MIEVFORMER_SINGLE_SLICE_CA,
        'single_slice': MIEVFORMER_SINGLE_SLICE_CA,
        'mievformer_ca': MIEVFORMER_SINGLE_SLICE_CA,
        'reference_probability_ca': MIEVFORMER_SINGLE_SLICE_CA,
        'multi': MIEVFORMER_MULTI_SLICE_CA,
        'multi_slice': MIEVFORMER_MULTI_SLICE_CA,
        'sample_conditional': MIEVFORMER_MULTI_SLICE_CA,
        'sample_conditioned': MIEVFORMER_MULTI_SLICE_CA,
        'sample_conditional_reference_probability_ca': MIEVFORMER_MULTI_SLICE_CA,
    }
    normalized = str(strategy).strip().lower()
    if normalized not in aliases:
        raise ValueError(
            'strategy must be auto, reference_probability_ca, or '
            'sample_conditional_reference_probability_ca'
        )
    resolved = aliases[normalized]
    if resolved == 'auto':
        resolved = (
            MIEVFORMER_MULTI_SLICE_CA
            if n_samples > 1
            else MIEVFORMER_SINGLE_SLICE_CA
        )
    if resolved == MIEVFORMER_MULTI_SLICE_CA and n_samples < 2:
        raise ValueError(
            'Sample-conditional CA requires a sample_key with at least two slices'
        )
    return resolved


def _default_mievformer_ca_reference_num(
    adata,
    strategy,
    sample_key=None,
    maximum=1000,
):
    maximum = min(int(maximum), int(adata.n_obs))
    if strategy == MIEVFORMER_SINGLE_SLICE_CA:
        if maximum < 2:
            raise ValueError('Mievformer CA requires at least two cells')
        return maximum
    effective_sample_key, sample_order = _mievformer_sample_context(
        adata,
        sample_key=sample_key,
    )
    batches = adata.obs[effective_sample_key].astype(str).to_numpy()
    counts = {
        sample: int(np.count_nonzero(batches == sample))
        for sample in sample_order
    }
    for reference_num in range(maximum, len(sample_order) - 1, -1):
        quotas = scca.equal_sample_quotas(sample_order, reference_num)
        if all(counts[sample] >= quotas[sample] for sample in sample_order):
            if reference_num < 3:
                break
            return reference_num
    raise ValueError(
        'Sample-conditional CA needs at least three balanced references across samples'
    )


def _require_mievformer_e2w(model):
    if model is None:
        raise ValueError(
            'A batch-corrected Mievformer model is required for multi-slice '
            'sample-conditional CA; ordinary CA is not used as a fallback.'
        )
    distributor = getattr(model, 'distributor', None)
    e2w = getattr(distributor, 'e2w', None)
    if not isinstance(e2w, torch.nn.Module):
        raise ValueError('The Mievformer model does not expose distributor.e2w')
    return e2w


def fit_sample_conditional_reference_probability_ca(
    adata,
    model,
    sample_key,
    reference_num=1000,
    reference_seed=0,
    n_components='auto',
    dimension_reference_seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    candidate_dimensions=(5, 10, 20, 40),
    max_spectrum_components=40,
    max_fit_cells=50000,
    fit_seed=0,
    e_key=None,
    batch_one_hot_key='batch_one_hot',
    w_e_key='w_e',
    w_z_key='w_z',
    b_z_key='b_z',
    query_batch_size=2048,
    device=None,
):
    """Fit reusable multi-slice, sample-conditional Mievformer CA."""
    _require_reference_probability_ca_inputs(
        adata,
        w_e_key=w_e_key,
        w_z_key=w_z_key,
        b_z_key=b_z_key,
    )
    effective_sample_key, sample_order = _mievformer_sample_context(
        adata,
        sample_key=sample_key,
    )
    if len(sample_order) < 2:
        raise ValueError('Sample-conditional CA requires at least two slices')
    if batch_one_hot_key not in adata.obsm:
        raise ValueError(
            f"obsm[{batch_one_hot_key!r}] is required for multi-slice "
            'sample-conditional CA; ordinary CA is not used as a fallback.'
        )
    raw_e_key = _mievformer_raw_e_key(adata, e_key=e_key)
    raw_e = np.asarray(adata.obsm[raw_e_key], dtype=np.float32)
    batch_one_hot = np.asarray(adata.obsm[batch_one_hot_key], dtype=np.float32)
    batches = adata.obs[effective_sample_key].astype(str).to_numpy()
    one_hot_mapping = scca.derive_sample_one_hot_mapping(
        batches,
        batch_one_hot,
        expected_order=sample_order,
    )
    e2w = _require_mievformer_e2w(model)
    reference_num = int(reference_num)
    if max_fit_cells is None:
        n_fit = int(adata.n_obs)
    else:
        n_fit = min(int(max_fit_cells), int(adata.n_obs))
    if n_fit < 3:
        raise ValueError('At least three fitting cells are required')
    if n_fit == adata.n_obs:
        fit_indices = np.arange(adata.n_obs, dtype=np.int64)
    else:
        fit_indices = np.sort(
            np.random.default_rng(int(fit_seed)).choice(
                adata.n_obs,
                size=n_fit,
                replace=False,
            )
        ).astype(np.int64)

    with _module_on_device(e2w, device) as resolved_device:
        reproduction = scca.verify_observed_w_e(
            raw_e,
            batch_one_hot,
            adata.obsm[w_e_key],
            e2w,
            batch_size=int(query_batch_size),
            device=resolved_device,
        )
        ca_model = scca.fit_sample_conditional_ca(
            raw_e[fit_indices],
            adata.obsm[w_z_key],
            adata.obsm[b_z_key],
            batches,
            sample_order=sample_order,
            e2w=e2w,
            sample_one_hot=one_hot_mapping,
            reference_num=reference_num,
            reference_seed=int(reference_seed),
            n_components=n_components,
            dimension_reference_seeds=dimension_reference_seeds,
            candidate_dimensions=candidate_dimensions,
            max_spectrum_components=int(max_spectrum_components),
            query_batch_size=int(query_batch_size),
            fit_indices=fit_indices,
            device=resolved_device,
        )
    ca_model.update(
        {
            'strategy': MIEVFORMER_MULTI_SLICE_CA,
            'sample_key': effective_sample_key,
            'n_samples': len(sample_order),
            'e_key': raw_e_key,
            'batch_one_hot_key': str(batch_one_hot_key),
            'w_e_key': str(w_e_key),
            'w_z_key': str(w_z_key),
            'b_z_key': str(b_z_key),
            'reference_obs_names': np.asarray(
                adata.obs_names.astype(str)[ca_model['reference_indices']],
                dtype=str,
            ),
            'n_fit_cells': int(n_fit),
            'fit_seed': int(fit_seed),
            'distributor_reproduction': reproduction,
        }
    )
    return ca_model


def transform_sample_conditional_reference_probability_ca(
    adata,
    ca_model,
    model,
    e_key=None,
    query_batch_size=None,
    device=None,
    return_audit=False,
):
    """Project cells using a frozen sample-conditional Mievformer CA model."""
    if ca_model.get('method') != 'mievformer_sample_conditional_reference_probability_ca':
        raise ValueError('The supplied CA model is not sample-conditional')
    if e_key is None:
        e_key = ca_model.get('e_key')
    raw_e = _mievformer_raw_e(adata, e_key=e_key)
    if query_batch_size is None:
        query_batch_size = int(ca_model.get('query_batch_size', 2048))
    e2w = _require_mievformer_e2w(model)
    with _module_on_device(e2w, device) as resolved_device:
        scores, audit = scca.transform_sample_conditional_ca(
            raw_e,
            ca_model,
            e2w=e2w,
            batch_size=int(query_batch_size),
            device=resolved_device,
        )
    if return_audit:
        return scores, audit
    return scores


def fit_mievformer_ca(
    adata,
    model=None,
    sample_key=None,
    strategy='auto',
    reference_num=None,
    reference_seed=0,
    n_components='auto',
    dimension_reference_seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    candidate_dimensions=(5, 10, 20, 40),
    max_spectrum_components=40,
    max_fit_cells=50000,
    fit_seed=0,
    e_key=None,
    batch_one_hot_key='batch_one_hot',
    w_e_key='w_e',
    w_z_key='w_z',
    b_z_key='b_z',
    query_batch_size=2048,
    device=None,
):
    """Fit the standard Mievformer CA selected by observed slice count."""
    resolved_strategy = resolve_mievformer_ca_strategy(
        adata,
        sample_key=sample_key,
        strategy=strategy,
    )
    effective_sample_key, sample_order = _mievformer_sample_context(
        adata,
        sample_key=sample_key,
    )
    if reference_num is None:
        reference_num = _default_mievformer_ca_reference_num(
            adata,
            resolved_strategy,
            sample_key=effective_sample_key,
        )
    common_kwargs = {
        'reference_num': int(reference_num),
        'reference_seed': int(reference_seed),
        'n_components': n_components,
        'dimension_reference_seeds': dimension_reference_seeds,
        'candidate_dimensions': candidate_dimensions,
        'max_spectrum_components': int(max_spectrum_components),
        'max_fit_cells': max_fit_cells,
        'fit_seed': int(fit_seed),
        'w_e_key': w_e_key,
        'w_z_key': w_z_key,
        'b_z_key': b_z_key,
        'query_batch_size': int(query_batch_size),
        'device': device,
    }
    if resolved_strategy == MIEVFORMER_SINGLE_SLICE_CA:
        ca_model = fit_reference_probability_ca(adata, **common_kwargs)
        ca_model = dict(ca_model)
        ca_model.update(
            {
                'strategy': MIEVFORMER_SINGLE_SLICE_CA,
                'sample_key': '' if effective_sample_key is None else effective_sample_key,
                'n_samples': max(1, len(sample_order)),
                'standard_scope': 'single_slice',
            }
        )
        return ca_model
    return fit_sample_conditional_reference_probability_ca(
        adata,
        model,
        effective_sample_key,
        e_key=e_key,
        batch_one_hot_key=batch_one_hot_key,
        **common_kwargs,
    )


def transform_mievformer_ca(
    adata,
    ca_model,
    model=None,
    e_key=None,
    query_batch_size=None,
    device=None,
    return_audit=False,
):
    """Project cells with either supported Mievformer CA artifact."""
    strategy = ca_model.get('strategy')
    method = ca_model.get('method')
    if (
        strategy == MIEVFORMER_MULTI_SLICE_CA
        or method == 'mievformer_sample_conditional_reference_probability_ca'
    ):
        return transform_sample_conditional_reference_probability_ca(
            adata,
            ca_model,
            model,
            e_key=e_key,
            query_batch_size=query_batch_size,
            device=device,
            return_audit=return_audit,
        )
    scores = transform_reference_probability_ca(
        adata,
        ca_model,
        query_batch_size=query_batch_size,
        device=device,
    )
    if return_audit:
        return scores, {}
    return scores


def _set_mievformer_ca_as_default(adata, scores, ca_model, feature_key):
    raw_e_key = _mievformer_raw_e_key(adata, e_key=ca_model.get('e_key'))
    if raw_e_key == 'e':
        adata.obsm[MIEVFORMER_RAW_E_KEY] = np.asarray(
            adata.obsm['e'], dtype=np.float32
        ).copy()
        raw_e_key = MIEVFORMER_RAW_E_KEY
        ca_model['e_key'] = raw_e_key
    adata.obsm['e'] = np.asarray(scores, dtype=np.float32).copy()
    adata.uns[MIEVFORMER_DEFAULT_REPRESENTATION_KEY] = {
        'default_embedding_key': 'e',
        'ca_feature_key': str(feature_key),
        'raw_embedding_key': raw_e_key,
        'strategy': str(ca_model['strategy']),
        'sample_key': str(ca_model.get('sample_key', '')),
        'n_samples': int(ca_model.get('n_samples', 1)),
        'selected_n_components': int(ca_model['selected_n_components']),
        'selection_policy': 'single_slice_ca_multi_slice_sample_conditional_ca',
        'raw_embedding_retained': True,
        'silent_fallback_used': False,
    }


def add_mievformer_ca_features(
    adata,
    model=None,
    sample_key=None,
    strategy='auto',
    feature_key=MIEVFORMER_CA_KEY,
    cell_rep_key='X_pca',
    set_as_default=True,
    copy=False,
    **fit_kwargs,
):
    """Add the benchmark-supported standard Mievformer representation.

    Single-slice data use ordinary reference-probability CA.  Multi-slice data
    use sample-conditional CA and require the exact batch-corrected model and
    persisted batch one-hot rows.  Missing multi-slice inputs raise an error;
    there is no ordinary-CA or distance-based fallback.
    """
    if copy:
        adata = adata.copy()
    if feature_key == 'e':
        raise ValueError("feature_key='e' is reserved for the standard-view alias")
    weight_keys = [
        fit_kwargs.get('w_e_key', 'w_e'),
        fit_kwargs.get('w_z_key', 'w_z'),
        fit_kwargs.get('b_z_key', 'b_z'),
    ]
    if not all(key in adata.obsm for key in weight_keys):
        if model is None:
            raise ValueError(
                f'Missing Mievformer weight arrays {weight_keys}; provide the '
                'exact model or run add_wb_ez. No fallback is used.'
            )
        if weight_keys != ['w_e', 'w_z', 'b_z']:
            raise ValueError('Custom weight keys cannot be generated by add_wb_ez')
        adata = add_wb_ez(
            adata,
            model,
            cell_rep_key=cell_rep_key,
            e_key=fit_kwargs.get('e_key'),
        )
    ca_model = fit_mievformer_ca(
        adata,
        model=model,
        sample_key=sample_key,
        strategy=strategy,
        **fit_kwargs,
    )
    scores, transform_audit = transform_mievformer_ca(
        adata,
        ca_model,
        model=model,
        e_key=fit_kwargs.get('e_key'),
        query_batch_size=fit_kwargs.get('query_batch_size'),
        device=fit_kwargs.get('device'),
        return_audit=True,
    )
    ca_model['transform_probability_audit'] = transform_audit
    adata.obsm[feature_key] = np.asarray(scores, dtype=np.float32)
    if set_as_default:
        _set_mievformer_ca_as_default(adata, scores, ca_model, feature_key)
    adata.uns[feature_key] = ca_model
    return adata, ca_model


def project_mievformer_ca_features(
    adata,
    ca_model,
    model=None,
    feature_key=MIEVFORMER_CA_KEY,
    set_as_default=True,
    e_key=None,
    query_batch_size=None,
    device=None,
    copy=False,
):
    """Add a standard Mievformer CA view using a frozen fitted artifact."""
    if copy:
        adata = adata.copy()
    scores, transform_audit = transform_mievformer_ca(
        adata,
        ca_model,
        model=model,
        e_key=e_key,
        query_batch_size=query_batch_size,
        device=device,
        return_audit=True,
    )
    stored_model = dict(ca_model)
    stored_model['transform_probability_audit'] = transform_audit
    adata.obsm[feature_key] = np.asarray(scores, dtype=np.float32)
    if set_as_default:
        _set_mievformer_ca_as_default(adata, scores, stored_model, feature_key)
    adata.uns[feature_key] = stored_model
    return adata
