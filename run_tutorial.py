"""Self-contained single-slice Mievformer tutorial."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

import mievformer as mf


def make_synthetic_spatial_data(seed=7):
    rng = np.random.default_rng(seed)
    side_x, side_y = 12, 10
    n_cells = side_x * side_y
    spatial = np.stack(
        np.meshgrid(np.arange(side_x), np.arange(side_y)), axis=-1
    ).reshape(-1, 2)
    spatial = spatial + rng.normal(scale=0.08, size=spatial.shape)
    cell_type_index = np.digitize(spatial[:, 0], bins=[4.0, 8.0])
    cell_types = np.asarray(['TypeA', 'TypeB', 'TypeC'])[cell_type_index]

    means = np.full((n_cells, 30), 1.2)
    for group in range(3):
        means[cell_type_index == group, group * 5:(group + 1) * 5] += 3.0
    counts = rng.poisson(means).astype(np.float32)
    centered = counts - counts.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)

    adata = sc.AnnData(
        counts,
        obs=pd.DataFrame(
            {'cell_type': cell_types},
            index=[f'cell_{index:03d}' for index in range(n_cells)],
        ),
    )
    adata.var_names = [f'Gene_{index:02d}' for index in range(adata.n_vars)]
    adata.obsm['spatial'] = spatial.astype(np.float32)
    adata.obsm['X_pca'] = (u[:, :10] * singular_values[:10]).astype(np.float32)
    adata.uns['log1p'] = {'base': None}
    return adata


np.random.seed(7)
torch.manual_seed(7)
adata = make_synthetic_spatial_data()
print(f'Created synthetic data with {adata.n_obs} cells and {adata.n_vars} genes.')

model_path = 'tutorial_model.pth'
adata = mf.optimize_nicheformer(
    adata,
    model_path=model_path,
    ngpu=1 if torch.cuda.is_available() else 0,
    max_epochs=1,
    batch_size=32,
    latent_dim=6,
    neighbor_num=5,
    ca_reference_num=30,
    ca_n_components=5,
    ca_device='cuda' if torch.cuda.is_available() else 'cpu',
    niche_n_neighbors=10,
    random_state=7,
)
print('Optimization complete.')
print('Standard representation:', adata.uns['mievformer_default_representation']['strategy'])

# optimize_nicheformer already computes these weights. Calling the public
# helper again demonstrates how an existing model can be reloaded safely.
adata = mf.calculate_wb_ez(
    adata,
    model_path,
    neighbor_num=5,
    latent_dim=6,
)

adata = mf.calculate_niche_density_ratio(
    adata,
    ref_num=30,
    stratify_key='leiden_e',
)
adata = mf.calculate_niche_cluster_membership(adata)
print('Niche density ratio and cluster membership computed.')

target_group = 'TypeA'
print(f'Estimating density for: {target_group}')
adata = mf.estimate_population_density(
    adata,
    group=target_group,
    cluster_key='cell_type',
    max_cell_num=30,
)
print(f"Density estimated. Added '{target_group}_density' to adata.obs.")

corrs = mf.analyze_density_correlation(
    adata,
    density_col=f'{target_group}_density',
    file_path='density_correlation.png',
)
print('Top 5 correlated genes:')
print(corrs.nlargest(5))

adata = mf.analyze_niche_membership(
    adata,
    n_clusters=3,
    file_path='niche_composition_clustermap.png',
)
print('Niche membership analysis complete.')

sc.pl.umap(adata, color=['leiden_e', 'cell_type'], show=False)
plt.savefig('ca_umap.png', bbox_inches='tight')
plt.close('all')
print('Tutorial complete.')
