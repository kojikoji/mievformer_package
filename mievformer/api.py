import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from . import workflow as wl
import scipy

def optimize_nicheformer(
    adata,
    model_path,
    ngpu=1,
    batch_size=512,
    max_epochs=1000,
    neighbor_num=100,
    latent_dim=20,
    kld_ld=0.05,
    pent_ld=0.05,
    dist_space='latent',
    cellrep_key='X_pca',
    batch_key=None,
    batch_correct='auto',
    representation_mode='auto',
    ca_reference_num=None,
    ca_n_components='auto',
    ca_reference_seed=0,
    ca_device=None,
    niche_n_neighbors=15,
    leiden_resolution=0.5,
    umap_min_dist=0.1,
    random_state=0,
):
    """Train Mievformer and create its standard niche representation.

    By default, single-slice data use reference-probability correspondence
    analysis (CA), while data with multiple values in batch_key use
    sample-conditional CA. The selected CA representation is stored in both
    obsm['reference_probability_ca'] and obsm['e']; the raw model embedding
    is preserved in obsm['mievformer_raw_e']. Neighbors, UMAP, and
    obs['leiden_e'] are calculated from the standard obsm['e'].

    Set representation_mode='raw' only when reproducing the legacy
    raw-embedding workflow. Multi-slice standard analysis requires
    batch_correct='auto' (the default) or True.

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial data containing obsm['spatial'] and the representation
        selected by cellrep_key.
    model_path : path-like
        Destination for the trained state dictionary.
    batch_key : str, optional
        obs column identifying spatial slices or samples.
    batch_correct : bool or {'auto'}, default 'auto'
        auto enables sample conditioning when batch_key has multiple values.
    representation_mode : {'auto', 'raw'}, default 'auto'
        Standard CA selection or explicit legacy raw-embedding mode.
    ca_reference_num : int, optional
        Number of CA reference cells. The default is adaptive up to 1000.
    ca_n_components : int or {'auto'}, default 'auto'
        CA dimension. Automatic selection uses the mean relative eigengap.
    niche_n_neighbors : int, default 15
        Number of neighbors for the CA-based Scanpy graph. This is independent
        of neighbor_num, which controls spatial context during training.

    Returns
    -------
    anndata.AnnData
        A copy containing raw and standard representations, score-function
        weights, UMAP coordinates, niche clusters, and provenance metadata.
    """
    from . import pipeline

    return pipeline.optimize(
        adata,
        model_path,
        ngpu=ngpu,
        batch_size=batch_size,
        max_epochs=max_epochs,
        neighbor_num=neighbor_num,
        latent_dim=latent_dim,
        kld_ld=kld_ld,
        pent_ld=pent_ld,
        dist_space=dist_space,
        cellrep_key=cellrep_key,
        batch_key=batch_key,
        batch_correct=batch_correct,
        representation_mode=representation_mode,
        ca_reference_num=ca_reference_num,
        ca_n_components=ca_n_components,
        ca_reference_seed=ca_reference_seed,
        ca_device=ca_device,
        niche_n_neighbors=niche_n_neighbors,
        leiden_resolution=leiden_resolution,
        umap_min_dist=umap_min_dist,
        random_state=random_state,
    )


def calculate_wb_ez(
    adata,
    model_path,
    batch_key=None,
    batch_correct='auto',
    neighbor_num=100,
    latent_dim=20,
    cellrep_key='X_pca',
):
    """Calculate the Mievformer score-function weights.

    The function adds obsm['w_e'], obsm['w_z'], and obsm['b_z']. If CA is
    already the default obsm['e'], the preserved obsm['mievformer_raw_e'] is
    used as the distributor input.
    """
    from . import pipeline

    return pipeline.calculate_weights(
        adata,
        model_path,
        batch_key=batch_key,
        batch_correct=batch_correct,
        neighbor_num=neighbor_num,
        latent_dim=latent_dim,
        cellrep_key=cellrep_key,
    )


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
