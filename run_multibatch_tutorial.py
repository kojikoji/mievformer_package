"""Self-contained multi-batch Mievformer tutorial."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

import mievformer as mf


def make_synthetic_multibatch_data(seed=11):
    rng = np.random.default_rng(seed)
    cells_per_slice = 60
    n_cells = 2 * cells_per_slice
    slice_labels = np.repeat(['slice_A', 'slice_B'], cells_per_slice)

    local_spatial = np.vstack(
        [rng.uniform(0, 8, size=(cells_per_slice, 2)) for _ in range(2)]
    )
    cell_type_index = (
        (local_spatial[:, 0] > 4).astype(int)
        + (local_spatial[:, 1] > 5).astype(int)
    )
    cell_type_index = np.clip(cell_type_index, 0, 2)
    cell_types = np.asarray(['TypeA', 'TypeB', 'TypeC'])[cell_type_index]

    means = np.full((n_cells, 30), 1.0)
    for group in range(3):
        means[cell_type_index == group, group * 5:(group + 1) * 5] += 3.0
    means[slice_labels == 'slice_B', 15:20] += 1.0
    counts = rng.poisson(means).astype(np.float32)
    centered = counts - counts.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)

    # Deliberately interleave slices. The public pipeline must restore this
    # exact order after its internal contiguous-batch computation.
    order = np.ravel(
        np.column_stack(
            [np.arange(cells_per_slice), np.arange(cells_per_slice, n_cells)]
        )
    )
    obs = pd.DataFrame(
        {
            'sample': slice_labels[order],
            'cell_type': cell_types[order],
        },
        index=[f'cell_{index:03d}' for index in range(n_cells)],
    )
    adata = sc.AnnData(counts[order], obs=obs)
    adata.var_names = [f'Gene_{index:02d}' for index in range(adata.n_vars)]
    adata.obsm['spatial'] = local_spatial[order].astype(np.float32)
    pca = u[:, :10] * singular_values[:10]
    adata.obsm['X_pca'] = pca[order].astype(np.float32)
    return adata


np.random.seed(11)
torch.manual_seed(11)
adata = make_synthetic_multibatch_data()
original_obs_names = adata.obs_names.copy()
print(f'Created synthetic multi-batch data with {adata.n_obs} cells.')

adata = mf.optimize_nicheformer(
    adata,
    model_path='multibatch_tutorial_model.pth',
    ngpu=1 if torch.cuda.is_available() else 0,
    max_epochs=1,
    batch_size=32,
    latent_dim=6,
    neighbor_num=5,
    batch_key='sample',
    batch_correct='auto',
    ca_reference_num=30,
    ca_n_components=5,
    ca_device='cuda' if torch.cuda.is_available() else 'cpu',
    niche_n_neighbors=10,
    random_state=11,
)

assert adata.obs_names.equals(original_obs_names)
assert adata.uns['mievformer_default_representation']['strategy'] == (
    'sample_conditional_reference_probability_ca'
)
print('Multi-batch optimization complete.')
print('Original cell order preserved.')
print('Strategy:', adata.uns['mievformer_default_representation']['strategy'])
print('Batch order:', adata.uns['mievformer_batch_contract']['batch_order'])

sc.pl.umap(adata, color=['sample', 'cell_type', 'leiden_e'], show=False)
plt.savefig('multibatch_ca_umap.png', bbox_inches='tight')
plt.close('all')
adata.write_h5ad('multibatch_tutorial_result.h5ad')
print('Multi-batch tutorial complete.')
