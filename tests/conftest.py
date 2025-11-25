
import pytest
import numpy as np
import pandas as pd
import anndata
import torch
import os
from mievformer import nicheformer as nf

@pytest.fixture
def dummy_adata():
    n_obs = 100
    n_vars = 50
    X = np.random.rand(n_obs, n_vars)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    obs['batch'] = np.random.choice(['b1', 'b2'], n_obs)
    obs['leiden_e'] = np.random.choice(['c1', 'c2', 'c3'], n_obs)
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.obsm['spatial'] = np.random.rand(n_obs, 2)
    adata.obsm['X_pca'] = np.random.rand(n_obs, 10)
    # Ensure log1p for some functions
    adata.uns['log1p'] = {'base': None}
    return adata

@pytest.fixture
def dummy_model_path(tmp_path):
    input_dim = 10
    latent_dim = 5
    model = nf.NicheFormer(input_dim=input_dim, latent_dim=latent_dim, train_ds=None, val_ds=None)
    model_path = tmp_path / "dummy_model.pt"
    torch.save(model.state_dict(), model_path)
    return str(model_path)

@pytest.fixture
def dummy_model_params():
    return {
        'input_dim': 10,
        'latent_dim': 5,
        'neighbor_num': 5
    }
