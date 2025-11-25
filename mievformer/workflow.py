import math
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from statsmodels.stats import multitest
from scipy import stats
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
from sklearn.neighbors import NearestNeighbors



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
    

def add_dist_across_cells(adata, model, output_mode='original', ref_num=1000):
    if 'batch_one_hot' in adata.obsm.keys():
        eb_mat = np.concatenate([adata.obsm['e'], adata.obsm['batch_one_hot']], axis=1)
        eds = torch.utils.data.TensorDataset(torch.tensor(eb_mat).float())
    else:
        eds = torch.utils.data.TensorDataset(torch.tensor(adata.obsm['e']).float())
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

def add_wb_ez(adata, model, cell_rep_key='X_pca'):
    if 'batch_one_hot' in adata.obsm.keys():
        eb_mat = np.concatenate([adata.obsm['e'], adata.obsm['batch_one_hot']], axis=1)
        ezds = torch.utils.data.TensorDataset(torch.tensor(eb_mat).float(), torch.tensor(adata.obsm[cell_rep_key]).float())
    else:
        ezds = torch.utils.data.TensorDataset(torch.tensor(adata.obsm['e']).float(), torch.tensor(adata.obsm[cell_rep_key]).float())
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



def loading_pre_trained_model(model_path, adata, model_id_dict={}):
    latent_dim = int(model_id_dict.get('ldim', 20))
    cellrep_key = model_id_dict.get('crkey', 'X_pca')
    batch_correct = model_id_dict.get('bcorr', 'false') =='true'
    nlayers = int(model_id_dict.get('nlayers', 3))
    nheads = int(model_id_dict.get('nheads', 1))
    model = nf.NicheFormer(input_dim=adata.obsm[cellrep_key].shape[1], latent_dim=latent_dim, train_ds=None, val_ds=None, batch_correct=batch_correct, num_layers=nlayers, head_num=nheads)
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
    data_loader = DataLoader(dataset, batch_size=1024, num_workers=12, pin_memory=True)
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



def calculate_spatial_distribution(adata, ref_niche_num, stratify_key='leiden_e', min_ratio=0.01, ref_adata=None):
    """
    Perform calculation of spatial distributions across given number of niches.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    ref_niche_num : int
        The number of reference niches to use for caluculaiton of spatial distributions.
    
    Returns
    -------
    adata : AnnData
        The updated annotated data matrix with clustering information. The spatial distribution is stored in adata.obsm['dist_e']. The obs_names of reference niche point is stored in uns['dist_e']['ref_obs'].
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
        Number of reference niches used in spatial distribution calculation.
    
    Returns
    -------
    comm_strength_df : pd.DataFrame
        A matrix of communication strengths between niche clusters.
    """
    if 'dist_e_agg' not in adata.obsm.keys():
        raise ValueError("dist_e_agg is not found in adata.obsm. Please run post_process_nicheformer first.")
    
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


def aggregate_dist_e(adata, group_key='leiden_e'):
    """
    Perform aggregation of spatial distributions based on a given group key.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    group_key : str
        The key for the grouping information in adata.obs.
    
    Returns
    -------
    adata : AnnData
        The updated annotated data matrix with aggregated spatial distributions.
    """
    if 'dist_e' not in adata.obsm.keys():
        adata = calculate_spatial_distribution(adata, ref_niche_num=1000, stratify_key=group_key)
        print("Spatial distribution calculated and stored in adata.obsm['dist_e'].")
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

def cluster_niche_composition(adata, n_clusters=15, use_rep='dist_e_agg', key_added='niche_composition_cluster'):
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

def visualize_niche_composition(adata, cluster_key='niche_composition_cluster', use_rep='dist_e_agg', file_path=None):
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

