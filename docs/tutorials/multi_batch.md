# Multi-Batch Analysis

Mievformer supports joint analysis of multiple spatial slices as a standard
workflow. Training is sample-conditioned, spatial neighbors are calculated
within each slice, and the final representation uses sample-conditional
reference-probability CA. The executable example is
run_multibatch_tutorial.py and does not download external data.

## Prepare the AnnData object

Concatenate slices into one AnnData object and provide an obs column that
identifies their source. Cells do not need to be grouped by sample; the API
orders them internally and restores the original order before returning.

~~~python
import numpy as np
import pandas as pd
import scanpy as sc

rng = np.random.default_rng(11)
cells_per_slice = 60
labels = np.repeat(['slice_A', 'slice_B'], cells_per_slice)
spatial = np.vstack([
    rng.uniform(0, 8, size=(cells_per_slice, 2)),
    rng.uniform(0, 8, size=(cells_per_slice, 2)),
])
counts = rng.poisson(1.5, size=(120, 30)).astype(np.float32)
u, s, _ = np.linalg.svd(
    counts - counts.mean(axis=0, keepdims=True), full_matrices=False
)

# Interleave the two slices to demonstrate order restoration.
order = np.ravel(np.column_stack([
    np.arange(cells_per_slice),
    np.arange(cells_per_slice, 2 * cells_per_slice),
]))
adata = sc.AnnData(
    counts[order],
    obs=pd.DataFrame(
        {'sample': labels[order]},
        index=[f'cell_{i:03d}' for i in range(120)],
    ),
)
adata.obsm['spatial'] = spatial[order].astype(np.float32)
adata.obsm['X_pca'] = (u[:, :10] * s[:10])[order].astype(np.float32)
~~~

Coordinates from different slices may use the same numeric range. Mievformer
never interprets them as adjacent when batch_key is supplied.

## Run sample-conditional Mievformer CA

~~~python
import torch
import mievformer as mf

original_order = adata.obs_names.copy()
adata = mf.optimize_nicheformer(
    adata,
    model_path='multibatch_model.pth',
    batch_key='sample',
    batch_correct='auto',
    ngpu=1 if torch.cuda.is_available() else 0,
    max_epochs=1,
    batch_size=32,
    neighbor_num=5,
    latent_dim=6,
    ca_reference_num=30,
    ca_n_components=5,
    niche_n_neighbors=10,
    random_state=11,
)

assert adata.obs_names.equals(original_order)
print(adata.uns['mievformer_default_representation']['strategy'])
# sample_conditional_reference_probability_ca
~~~

With production-sized data, omit ca_reference_num and ca_n_components to use
adaptive reference sampling and automatic CA dimension selection.

## Multi-batch output contract

In addition to the standard CA keys from the single-slice workflow, the result
contains:

| AnnData key | Meaning |
| --- | --- |
| obsm['batch_one_hot'] | Persisted sample conditioning used by the model |
| uns['mievformer_batch_contract'] | Sample key, first-observed order, and counts |
| uns['reference_probability_ca'] | Balanced references, CA basis, and probability audits |

Every sample receives a balanced reference quota and equal total probability
mass in sample-conditional CA. If the exact batch-corrected model, one-hot
mapping, or sample contract is unavailable, the API raises an error instead of
silently falling back to ordinary CA.

To visualize mixing and retained biology:

~~~python
sc.pl.umap(adata, color=['sample', 'cell_type', 'leiden_e'])
~~~

Setting representation_mode='raw' and batch_correct=False is available only
for explicit legacy or method-comparison runs; it is not the standard
multi-batch analysis.
