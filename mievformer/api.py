import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from . import nicheformer as nf
from . import workflow as wl
from . import prob_nmfae as pnf
from . import utils
import importlib
import scipy

def optimize_nicheformer(adata, model_path, ngpu=1, batch_size=512, max_epochs=1000, 
                         neighbor_num=100, latent_dim=20, kld_ld=0.05, pent_ld=0.05, 
                         dist_space='latent', cellrep_key='X_pca', batch_key=None, 
                         batch_correct=False):
    """
    Optimize the NicheFormer model using masked self-supervised learning.

    Mievformer learns microenvironmental representations by encoding the cellular states and spatial 
    configurations of neighboring cells using a Transformer-based architecture. It masks the central 
    cell position and maximizes the likelihood that the observed central cell state would be generated 
    from the inferred microenvironmental embedding.

    The training objective corresponds to the InfoNCE loss, maximizing the mutual information between 
    microenvironmental embeddings and their corresponding central cell states.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing spatial transcriptomics data.
        Must contain spatial coordinates in `adata.obsm['spatial']` and cell representations 
        (e.g., PCA) in `adata.obsm[cellrep_key]`.
    model_path : str
        Path to save the trained model checkpoint (.pt file).
    ngpu : int, optional
        Number of GPUs to use for training. Default is 1.
    batch_size : int, optional
        Batch size for training. Default is 512.
    max_epochs : int, optional
        Maximum number of epochs for training. Default is 1000.
    neighbor_num : int, optional
        Number of neighbors to consider for the microenvironmental context. Default is 100.
    latent_dim : int, optional
        Dimensionality of the latent microenvironmental embedding. Default is 20.
    kld_ld : float, optional
        Weight for the KL divergence loss term (if applicable). Default is 0.05.
    pent_ld : float, optional
        Weight for the entropy regularization term. Default is 0.05.
    dist_space : str, optional
        Space in which to compute distances ('latent' or other). Default is 'latent'.
    cellrep_key : str, optional
        Key in `adata.obsm` containing the cell state representations (e.g., 'X_pca'). 
        Default is 'X_pca'.
    batch_key : str, optional
        Key in `adata.obs` indicating batch information for batch correction/splitting. 
        Default is None.
    batch_correct : bool, optional
        Whether to perform batch correction during training. Default is False.

    Returns
    -------
    anndata.AnnData
        The input AnnData object updated with the following fields:
        - `obsm['e']`: Microenvironmental embeddings.
        - `obs['leiden_e']`: Leiden clusters of the microenvironmental embeddings.
    """
    adata = adata.copy()
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
            val_adata = wl.clip_center_adata(batch_adata, 0.1)
            train_adata = batch_adata[batch_adata.obs_names.isin(val_adata.obs_names) == False]
            train_adata_list.append(train_adata)
            val_adata_list.append(val_adata)
        train_adata = anndata.concat(train_adata_list)
        val_adata = anndata.concat(val_adata_list)
    else:
        val_adata = wl.clip_center_adata(adata, 0.1)
        train_adata = adata[adata.obs_names.isin(val_adata.obs_names) == False]

    ds_train = wl.adata2ds(train_adata, neighbor_num=neighbor_num, batch_key=batch_key)
    ds_val = wl.adata2ds(val_adata, neighbor_num=neighbor_num, batch_key=batch_key)
    
    # Setup trainer
    log_dir = os.path.dirname(model_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(dirpath=f'{log_dir}/ckpt')
        logger = TensorBoardLogger(save_dir=log_dir, version=1, name='lightning_logs')
    else:
        checkpoint_callback = ModelCheckpoint(dirpath='ckpt')
        logger = TensorBoardLogger(save_dir='.', version=1, name='lightning_logs')
    
    accelerator = "gpu" if torch.cuda.is_available() and ngpu > 0 else "cpu"
    devices = ngpu if accelerator == "gpu" else 1
    
    trainer = pl.Trainer(max_epochs=max_epochs, devices=devices, accelerator=accelerator,
                         callbacks=[EarlyStopping(monitor="val_loss", patience=20), checkpoint_callback],
                         reload_dataloaders_every_n_epochs=1, 
                         strategy='ddp_find_unused_parameters_true' if ngpu > 1 else 'auto', 
                         logger=logger)

    # Setup model
    model = nf.NicheFormer(input_dim=adata.obsm['nf_cellrep'].shape[1], latent_dim=latent_dim, 
                           train_ds=ds_train, val_ds=ds_val, kld_ld=kld_ld, pent_ld=pent_ld, 
                           dist_space=dist_space, batch_size=batch_size, batch_correct=batch_correct)
    
    # Train
    val_loader = DataLoader(ds_val, batch_size=batch_size, num_workers=0, pin_memory=True)
    trainer.fit(model, val_dataloaders=val_loader)
    
    # Load best model
    model = nf.NicheFormer.load_from_checkpoint(checkpoint_callback.best_model_path, 
                                                input_dim=adata.obsm['nf_cellrep'].shape[1], 
                                                latent_dim=latent_dim, train_ds=ds_train, 
                                                val_ds=ds_val, batch_correct=batch_correct)
    
    # Save model
    torch.save(model.state_dict(), model_path)
    
    # Post-processing: Compute embeddings
    ds = wl.adata2ds(adata, neighbor_num=neighbor_num, batch_key=batch_key)
    e, mu, sigma = wl.output_dist_params(ds, model)
    adata.obsm['e'] = e.numpy()
    
    # Clustering
    sc.pp.neighbors(adata, use_rep='e', n_neighbors=neighbor_num)
    sc.tl.leiden(adata, key_added='leiden_e')
    
    return adata

def calculate_wb_ez(adata, model_path, batch_key=None, neighbor_num=100, latent_dim=20, cellrep_key='X_pca'):
    """
    Calculate the embeddings required for the score function and add them to the AnnData object.

    The score function is defined as:

    .. math::
        s_{\\theta}(e_i, z_j) = w_e(e_i)^\\top w_z(z_j) + b_z(z_j)

    where :math:`w_e` and :math:`w_z` are neural networks mapping microenvironmental and cell-state 
    embeddings to a shared hidden dimension, and :math:`b_z` provides a cell-state-dependent bias.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    model_path : str
        Path to the trained model checkpoint.
    batch_key : str, optional
        Key in `adata.obs` indicating batch information. Default is None.
    neighbor_num : int, optional
        Number of neighbors used during training. Default is 100.
    latent_dim : int, optional
        Latent dimension of the model. Default is 20.
    cellrep_key : str, optional
        Key in `adata.obsm` containing the cell state representations. Default is 'X_pca'.

    Returns
    -------
    anndata.AnnData
        The input AnnData object updated with:
        - `obsm['w_e']`: Projected microenvironmental embeddings.
        - `obsm['w_z']`: Projected cell state embeddings.
        - `obsm['b_z']`: Cell state bias terms.
        - `obsm['e']`: Microenvironmental embeddings (if not already present).
    """
    model_id_dict = {
        'ldim': latent_dim,
        'nn': neighbor_num,
        'crkey': cellrep_key,
        'bkey': batch_key
    }
    model = wl.loading_pre_trained_model(model_path, adata, model_id_dict)
    
    if cellrep_key == 'X':
         adata.obsm['nf_cellrep'] = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    else:
         adata.obsm['nf_cellrep'] = adata.obsm[cellrep_key]
         
    if 'e' not in adata.obsm:
        ds = wl.adata2ds(adata, neighbor_num=neighbor_num, batch_key=batch_key)
        e, mu, sigma = wl.output_dist_params(ds, model)
        adata.obsm['e'] = e.numpy()
    
    wl.add_wb_ez(adata, model, cell_rep_key='nf_cellrep')
    return adata

def calculate_niche_density_ratio(adata, ref_num=1000, stratify_key='leiden_e', min_ratio=0.01, ref_adata=None):
    """
    Compute per-cell density ratios over a panel of reference niches.

    For each cell :math:`i` and reference niche :math:`j` drawn by stratified
    sampling on ``stratify_key``, the log density ratio is

    .. math::
        \\log r_{ij}
        = \\log p(e_j \\mid z_i) - \\log p(e_j)
        = (w_z(z_i)^\\top w_e(e_j) + b_z(z_i))
          - \\log \\sum_{k \\in \\mathrm{ref}}
            \\exp(w_z(z_k)^\\top w_e(e_j) + b_z(z_k)).

    The matrix is then softmax-normalized per cell over reference niches,
    so each row of ``adata.obsm['dist_e']`` is a probability distribution
    over the sampled reference niches that emphasizes niches whose
    environment becomes more likely under the cell's state than under the
    marginal.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing ``w_e``, ``w_z``, and ``b_z`` in ``obsm``
        (produced by :func:`calculate_wb_ez`).
    ref_num : int, optional
        Number of reference niches to sample. Default is 1000.
    stratify_key : str, optional
        Key in ``adata.obs`` to use for stratified sampling of reference niches.
        Default is 'leiden_e'.
    min_ratio : float, optional
        Clusters with frequency below this fraction are dropped from stratified
        sampling. Default is 0.01.
    ref_adata : anndata.AnnData, optional
        External reference. If ``None``, a subset of ``adata`` is used.

    Returns
    -------
    anndata.AnnData
        Updated with ``obsm['dist_e']`` (softmax-normalized density ratios of
        shape ``(n_cells, ref_num)``) and ``uns['dist_e']['ref_obs']`` (obs
        names of the sampled reference niches). The ``dist_e`` key name is
        preserved for backward compatibility with existing h5ad artifacts.
    """
    wl.calculate_niche_density_ratio(adata, ref_niche_num=ref_num, stratify_key=stratify_key, min_ratio=min_ratio, ref_adata=ref_adata)
    return adata

def calculate_niche_cluster_membership(adata, cluster_key='leiden_e'):
    """
    Aggregate per-cell density ratios into a soft membership over niche clusters.

    Averages the columns of ``adata.obsm['dist_e']`` within each value of
    ``adata.obs[cluster_key]`` (typically ``leiden_e`` niche clusters),
    yielding ``adata.obsm['dist_e_agg']`` of shape
    ``(n_cells, n_niche_clusters)``: entry ``[i, c]`` is the mean density
    ratio :math:`p(e \\mid z_i)/p(e)` evaluated at reference cells in cluster
    ``c``, interpretable as a soft assignment of cell ``i`` to niche cluster ``c``.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing ``obsm['dist_e']`` (see
        :func:`calculate_niche_density_ratio`). If absent, it is computed
        with defaults.
    cluster_key : str, optional
        Key in ``adata.obs`` containing niche cluster labels. Default is 'leiden_e'.

    Returns
    -------
    anndata.AnnData
        Updated with ``obsm['dist_e_agg']``: per-cell niche-cluster membership
        (columns are niche cluster labels). The ``dist_e_agg`` key name is
        preserved for backward compatibility with existing h5ad artifacts
        used by figure scripts.
    """
    wl.calculate_niche_cluster_membership(adata, group_key=cluster_key)
    return adata

def estimate_population_density(adata, group, cluster_key, max_cell_num=1000):
    """
    Estimate the density (existence probability) of a specific cell population in each microenvironment.

    By integrating :math:`P(z|e)` over all cell states belonging to a specific cell population, 
    this function obtains the density of that population in microenvironment :math:`e`.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    group : str
        The label of the cell population (e.g., a specific cell type) to estimate density for.
    cluster_key : str
        Key in `adata.obs` containing the cell type/cluster labels.
    max_cell_num : int, optional
        Maximum number of cells to sample from the group for density estimation. 
        Default is 1000.

    Returns
    -------
    anndata.AnnData
        The input AnnData object updated with a new column in `obs` (e.g., `{group}_density`)
        representing the estimated density of the specified population for each cell's microenvironment.
    """
    wl.estimate_population_density(adata, group, cluster_key, max_cell_num)
    return adata

def analyze_density_correlation(adata, density_col, gene_list=None, file_path=None):
    """
    Analyze the correlation between estimated cell population density and gene expression.

    This analysis helps identify gene expression signatures associated with colocalization 
    with specific cell populations. For example, identifying genes upregulated in tumor cells 
    when they colocalize with endothelial cells.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing expression data and the density column.
    density_col : str
        Name of the column in `adata.obs` containing the estimated density values.
    gene_list : list of str, optional
        List of genes to include in the correlation analysis. If None, uses all genes in `adata.var_names`.
    file_path : str, optional
        Path to save the visualization plot (bar plot of top/bottom correlated genes). 
        If None, the plot is not saved.

    Returns
    -------
    pandas.Series
        A Series containing the correlation coefficients for each gene, indexed by gene name.
    """
    if gene_list is None:
        gene_list = adata.var_names
    
    density = adata.obs[density_col].values
    # Ensure density is numeric
    density = pd.to_numeric(density, errors='coerce')
    
    # Check if X is sparse
    X = adata[:, gene_list].X
    
    if scipy.sparse.issparse(X):
        n = X.shape[0]
        d_mean = density.mean()
        d_std = density.std()
        
        # Gene stats
        means = np.array(X.mean(axis=0)).flatten()
        sq_means = np.array(X.power(2).mean(axis=0)).flatten()
        stds = np.sqrt(sq_means - means**2)
        
        # Covariance
        # X.T @ density
        covs = (X.T @ density) / n - means * d_mean
        
        corrs_val = covs / (stds * d_std + 1e-12)
        corrs = pd.Series(corrs_val, index=gene_list)
        
    else:
        df_exp = pd.DataFrame(X, index=adata.obs_names, columns=gene_list)
        corrs = df_exp.corrwith(pd.Series(density, index=adata.obs_names))
    
    if file_path:
        # Visualize top/bottom 10
        top10 = corrs.nlargest(10)
        bottom10 = corrs.nsmallest(10)
        plot_data = pd.concat([bottom10, top10]).sort_values()
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=plot_data.values, y=plot_data.index, palette="vlag", orient="h")
        plt.title(f'Correlation with {density_col} (Top/Bottom 10)')
        plt.xlabel('Correlation coefficient')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        
    return corrs

def analyze_niche_membership(adata, n_clusters=15, file_path=None):
    """
    Cluster cells by their niche-cluster membership vectors and visualize the result.

    Uses ``adata.obsm['dist_e_agg']`` (per-cell soft membership over niche
    clusters produced by :func:`calculate_niche_cluster_membership`) as the
    feature space, performs Ward hierarchical clustering to partition cells
    into ``n_clusters`` groups, and draws a clustermap of the membership
    matrix with row-color annotations.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing ``obsm['dist_e_agg']``.
    n_clusters : int, optional
        Number of cell clusters to form. Default is 15.
    file_path : str, optional
        Path to save the resulting clustermap image. If ``None``, the plot is
        not saved.

    Returns
    -------
    anndata.AnnData
        The input AnnData with ``obs['niche_composition_cluster']`` added
        (cell cluster labels). The ``niche_composition_cluster`` key name is
        preserved for backward compatibility with existing h5ad artifacts.
    """
    wl.cluster_cells_by_niche_membership(adata, n_clusters=n_clusters)
    wl.plot_niche_membership_clustermap(adata, file_path=file_path)
    return adata