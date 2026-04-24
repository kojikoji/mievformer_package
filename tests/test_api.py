import pytest
import numpy as np
import pandas as pd
import os
import torch
from mievformer import api
from scipy import sparse

def test_optimize_nicheformer(dummy_adata, tmp_path):
    # This function trains a model, so it might be slow. 
    # We'll use very few epochs and small data.
    model_path = str(tmp_path / "optimized_model.pt")
    
    ngpu = 1 if torch.cuda.is_available() else 0
    
    # Mocking or using small parameters
    adata = api.optimize_nicheformer(
        dummy_adata, 
        model_path, 
        ngpu=ngpu, 
        batch_size=10, 
        max_epochs=1, 
        neighbor_num=5, 
        latent_dim=5,
        cellrep_key='X_pca'
    )
    
    assert os.path.exists(model_path)
    assert 'e' in adata.obsm
    assert 'mu' in adata.obsm
    assert 'sigma' in adata.obsm
    assert 'leiden_e' in adata.obs # optimize_nicheformer adds this

def test_calculate_wb_ez(dummy_adata, dummy_model_path, dummy_model_params):
    latent_dim = dummy_model_params['latent_dim']
    neighbor_num = dummy_model_params['neighbor_num']
    
    adata = api.calculate_wb_ez(
        dummy_adata, 
        dummy_model_path, 
        neighbor_num=neighbor_num, 
        latent_dim=latent_dim, 
        cellrep_key='X_pca'
    )
    
    assert 'w_e' in adata.obsm
    assert 'w_z' in adata.obsm
    assert 'b_z' in adata.obsm
    # 'e' should be calculated if not present
    assert 'e' in adata.obsm 

def test_calculate_niche_density_ratio(dummy_adata, dummy_model_path, dummy_model_params):
    # calculate_niche_density_ratio requires w_z, b_z which come from calculate_wb_ez
    # So we run calculate_wb_ez first
    latent_dim = dummy_model_params['latent_dim']
    neighbor_num = dummy_model_params['neighbor_num']

    adata = api.calculate_wb_ez(
        dummy_adata,
        dummy_model_path,
        neighbor_num=neighbor_num,
        latent_dim=latent_dim,
        cellrep_key='X_pca'
    )

    adata = api.calculate_niche_density_ratio(adata, ref_num=10, stratify_key='leiden_e')

    # writes softmax-normalized density ratios to obsm['dist_e'] and ref obs names to uns['dist_e']
    assert 'dist_e' in adata.obsm
    assert 'dist_e' in adata.uns

def test_calculate_niche_cluster_membership(dummy_adata, dummy_model_path, dummy_model_params):
    # Setup prerequisites
    latent_dim = dummy_model_params['latent_dim']
    neighbor_num = dummy_model_params['neighbor_num']
    api.calculate_wb_ez(
        dummy_adata,
        dummy_model_path,
        neighbor_num=neighbor_num,
        latent_dim=latent_dim,
        cellrep_key='X_pca'
    )

    # Ensure cluster key exists (added in conftest)
    cluster_key = 'leiden_e'

    adata = api.calculate_niche_cluster_membership(dummy_adata, cluster_key=cluster_key)

    assert 'dist_e_agg' in adata.obsm

def test_estimate_population_density(dummy_adata, dummy_model_path, dummy_model_params):
    # Setup prerequisites
    latent_dim = dummy_model_params['latent_dim']
    neighbor_num = dummy_model_params['neighbor_num']
    api.calculate_wb_ez(
        dummy_adata, 
        dummy_model_path, 
        neighbor_num=neighbor_num, 
        latent_dim=latent_dim, 
        cellrep_key='X_pca'
    )
    
    group = 'c1'
    cluster_key = 'leiden_e'
    
    # Ensure at least one cell is in c1
    if 'c1' not in dummy_adata.obs[cluster_key].values:
        dummy_adata.obs.iloc[0, dummy_adata.obs.columns.get_loc(cluster_key)] = 'c1'

    adata = api.estimate_population_density(dummy_adata, group=group, cluster_key=cluster_key, max_cell_num=10)
    
    assert f'{group}_density' in adata.obs

def test_analyze_density_correlation(dummy_adata):
    density_col = 'density'
    dummy_adata.obs[density_col] = np.random.rand(dummy_adata.shape[0])
    
    corrs = api.analyze_density_correlation(dummy_adata, density_col=density_col)
    
    assert len(corrs) == dummy_adata.shape[1]
    assert isinstance(corrs, pd.Series)

def test_analyze_density_correlation_sparse(dummy_adata):
    density_col = 'density'
    dummy_adata.obs[density_col] = np.random.rand(dummy_adata.shape[0])
    
    # Make X sparse
    dummy_adata.X = sparse.csr_matrix(dummy_adata.X)
    
    corrs = api.analyze_density_correlation(dummy_adata, density_col=density_col)
    
    assert len(corrs) == dummy_adata.shape[1]
    assert isinstance(corrs, pd.Series)

def test_analyze_niche_membership(dummy_adata, dummy_model_path, dummy_model_params, tmp_path):
    # Setup prerequisites
    latent_dim = dummy_model_params['latent_dim']
    neighbor_num = dummy_model_params['neighbor_num']
    api.calculate_wb_ez(
        dummy_adata,
        dummy_model_path,
        neighbor_num=neighbor_num,
        latent_dim=latent_dim,
        cellrep_key='X_pca'
    )

    # Need dist_e_agg
    cluster_key = 'leiden_e'
    # Ensure cluster key exists
    if cluster_key not in dummy_adata.obs:
        dummy_adata.obs[cluster_key] = np.random.choice(['c1', 'c2'], dummy_adata.shape[0])

    api.calculate_niche_cluster_membership(dummy_adata, cluster_key=cluster_key)

    file_path = str(tmp_path / "niche_composition.png")

    adata = api.analyze_niche_membership(dummy_adata, n_clusters=3, file_path=file_path)

    assert 'niche_composition_cluster' in adata.obs
    assert os.path.exists(file_path)
