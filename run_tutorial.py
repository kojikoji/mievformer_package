import mievformer as mf
import scanpy as sc
import numpy as np
import os
import torch
from mievformer import nicheformer as nf
from mievformer import workflow as wl
import matplotlib.pyplot as plt

# Path to the original data
original_data_path = "nichedynamics/data/20230629__230629pre/output-XETG00057__0003908__lung__20230629__073037/adata.h5ad"

# Load data
if os.path.exists(original_data_path):
    adata = sc.read_h5ad(original_data_path)
    print(f"Loaded data with {adata.shape[0]} cells and {adata.shape[1]} genes.")
    
    # Subsample to 10k cells for tutorial
    if adata.shape[0] > 10000:
        sc.pp.subsample(adata, n_obs=10000)
        print(f"Subsampled to {adata.shape[0]} cells.")
else:
    print(f"Data file not found at {original_data_path}. Please check the path.")
    # Create dummy data for testing if file not found
    adata = sc.AnnData(np.random.rand(10000, 100))
    adata.obsm['spatial'] = np.random.rand(10000, 2)
    adata.obsm['X_pca'] = np.random.rand(10000, 20)
    adata.obs['cell_type'] = np.random.choice(['TypeA', 'TypeB', 'TypeC'], 10000)
    # Ensure log1p for some functions
    adata.uns['log1p'] = {'base': None}
    # Ensure gene names are strings
    adata.var_names = [f"Gene_{i}" for i in range(100)]
    print("Created dummy data.")

# Define model path
model_path = "tutorial_model.pth"

# Optimize model
# We use a small number of epochs for demonstration purposes
adata = mf.optimize_nicheformer(
    adata, 
    model_path=model_path,
    max_epochs=10,  # Increase for real analysis
    batch_size=128,
    latent_dim=20,
    neighbor_num=100
)

print("Optimization complete.")

# Re-instantiate model for usage
# We need to recreate the dataset to get input dim
ds = wl.adata2ds(adata, neighbor_num=100)
model = nf.NicheFormer(
    input_dim=adata.obsm['nf_cellrep'].shape[1], 
    latent_dim=20, 
    train_ds=ds, 
    val_ds=ds # Dummy
)
model.load_state_dict(torch.load(model_path))

# Calculate wb_ez (required for spatial distribution)
adata = mf.calculate_wb_ez(adata, model_path)

# Compute per-cell niche density ratio
adata = mf.calculate_niche_density_ratio(adata)

# Aggregate into per-cell niche-cluster membership
adata = mf.calculate_niche_cluster_membership(adata)
print("Niche density ratio and cluster membership computed.")

# Estimate density for a specific group (e.g., the first one found)
target_group = adata.obs['cell_type'].unique()[0]
print(f"Estimating density for: {target_group}")

adata = mf.estimate_population_density(adata, group=target_group, cluster_key='cell_type')

print(f"Density estimated. Added '{target_group}_density' to adata.obs.")

# Analyze Density Correlation
density_col = f'{target_group}_density'
output_plot = "density_correlation.png"

corrs = mf.analyze_density_correlation(
    adata, 
    density_col=density_col, 
    file_path=output_plot
)

print("Top 5 correlated genes:")
print(corrs.nlargest(5))

# Analyze Niche Composition
# Ensure we have the necessary keys. 'leiden_e' is usually added by optimize_nicheformer or subsequent clustering.
if 'leiden_e' not in adata.obs:
    print("Running Leiden clustering on 'e' embedding...")
    sc.pp.neighbors(adata, use_rep='e')
    sc.tl.leiden(adata, key_added='leiden_e')

adata = mf.analyze_niche_membership(
    adata,
    n_clusters=3, # Number of cell clusters based on niche membership
    file_path='niche_composition_clustermap.png'
)

print("Niche membership analysis complete.")
