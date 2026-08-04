# Getting Started: Single-Slice Analysis

This tutorial is fully self-contained. It creates a deterministic synthetic
spatial transcriptomics dataset, trains Mievformer, and uses
reference-probability correspondence analysis (CA) for niche clustering and
UMAP. The complete executable version is provided as run_tutorial.py.

## 1. Create spatial data

Mievformer requires a cell representation in adata.obsm['X_pca'] and spatial
coordinates in adata.obsm['spatial']. The tutorial generator creates 120 cells
with three spatially organized cell types:

~~~python
import numpy as np
import pandas as pd
import scanpy as sc

rng = np.random.default_rng(7)
side_x, side_y = 12, 10
spatial = np.stack(
    np.meshgrid(np.arange(side_x), np.arange(side_y)), axis=-1
).reshape(-1, 2)
spatial = spatial + rng.normal(scale=0.08, size=spatial.shape)
cell_type_index = np.digitize(spatial[:, 0], bins=[4.0, 8.0])
cell_types = np.asarray(['TypeA', 'TypeB', 'TypeC'])[cell_type_index]

means = np.full((spatial.shape[0], 30), 1.2)
for group in range(3):
    means[cell_type_index == group, group * 5:(group + 1) * 5] += 3.0
counts = rng.poisson(means).astype(np.float32)
u, s, _ = np.linalg.svd(
    counts - counts.mean(axis=0, keepdims=True), full_matrices=False
)

adata = sc.AnnData(
    counts,
    obs=pd.DataFrame(
        {'cell_type': cell_types},
        index=[f'cell_{i:03d}' for i in range(spatial.shape[0])],
    ),
)
adata.var_names = [f'Gene_{i:02d}' for i in range(adata.n_vars)]
adata.obsm['spatial'] = spatial.astype(np.float32)
adata.obsm['X_pca'] = (u[:, :10] * s[:10]).astype(np.float32)
~~~

For real data, replace this block with your AnnData object while keeping the
same two obsm keys.

## 2. Train and calculate the standard CA view

optimize_nicheformer now performs model training, score-weight calculation,
CA, neighbors, UMAP, and Leiden clustering in one call:

~~~python
import torch
import mievformer as mf

adata = mf.optimize_nicheformer(
    adata,
    model_path='tutorial_model.pth',
    ngpu=1 if torch.cuda.is_available() else 0,
    max_epochs=1,          # Increase for real analysis
    batch_size=32,
    latent_dim=6,
    neighbor_num=5,       # Use a larger context for real datasets
    ca_reference_num=30,  # Omit for the adaptive default up to 1000
    ca_n_components=5,    # Omit to use automatic dimension selection
    niche_n_neighbors=10,
    random_state=7,
)
~~~

The standard representation contract is:

| AnnData key | Meaning |
| --- | --- |
| obsm['mievformer_raw_e'] | Raw neural-network embedding |
| obsm['reference_probability_ca'] | Reference-probability CA scores |
| obsm['e'] | Alias of the standard CA scores |
| obsm['X_umap'] | UMAP calculated from CA |
| obs['leiden_e'] | Leiden clusters calculated from CA |
| uns['mievformer_default_representation'] | Strategy and provenance |

Use representation_mode='raw' only for a controlled legacy comparison.

## 3. Downstream density-ratio analysis

The score weights are already available after optimization. The existing
density-ratio and niche-membership APIs therefore work directly:

~~~python
adata = mf.calculate_niche_density_ratio(
    adata, ref_num=30, stratify_key='leiden_e'
)
adata = mf.calculate_niche_cluster_membership(adata)

adata = mf.estimate_population_density(
    adata,
    group='TypeA',
    cluster_key='cell_type',
    max_cell_num=30,
)
corrs = mf.analyze_density_correlation(
    adata,
    density_col='TypeA_density',
    file_path='density_correlation.png',
)
adata = mf.analyze_niche_membership(
    adata,
    n_clusters=3,
    file_path='niche_composition_clustermap.png',
)
~~~

For multiple spatial samples, continue with the
[multi-batch tutorial](multi_batch.md).
