# Release Notes

## Version 0.3.0 (2026)

### Changed

- `optimize_nicheformer` now uses reference-probability correspondence analysis
  (CA) as the standard representation for neighbors, UMAP, and Leiden niche
  clustering.
- `obsm['e']` now aliases the standard CA scores. The former neural-network
  embedding is retained in `obsm['mievformer_raw_e']`.
- `batch_correct` now defaults to `auto`: it is enabled when `batch_key`
  identifies multiple samples.

### Added

- Single-slice reference-probability CA with deterministic reference sampling,
  automatic dimension selection, and reusable AnnData artifacts.
- Sample-conditional CA for multiple spatial slices, including balanced
  reference blocks, equal per-sample probability mass, and strict validation
  of the persisted one-hot mapping.
- Self-contained single-slice and multi-batch tutorials.
- `representation_mode='raw'` as an explicit legacy comparison mode.
- Direct `igraph` and `leidenalg` dependencies for the standard Leiden
  clustering step.

### Removed

- The experimental `mievformer.cci.binary_cci` module and its colocated test
  module have been removed.

### Compatibility

- Existing single-slice state dictionaries remain loadable.
- Existing h5ad files are not rewritten automatically. Re-run
  `optimize_nicheformer` to create the 0.3 standard representation contract.
- Multi-slice CA never silently falls back to ordinary CA when its exact model
  or conditioning metadata is missing.

## Version 0.2.0 (2026)

### Changed (Breaking — API rename for semantic clarity)

The API now names functions after the quantities they actually compute:

- `calculate_spatial_distribution` → `calculate_niche_density_ratio`. The function computes, per cell and per reference niche, a softmax over `log p(e|z) − log p(e)` — i.e. a density ratio p(e|z)/p(e) of the environment given the cell state versus its marginal. Docstring now states this explicitly.
- `aggregate_dist_e` → `calculate_niche_cluster_membership`. The output is a per-cell soft membership over niche clusters (which niche cluster each cell most likely belongs to).
- `analyze_niche_composition` → `analyze_niche_membership`.
- Internal helpers `cluster_niche_composition` / `visualize_niche_composition` renamed to `cluster_cells_by_niche_membership` / `plot_niche_membership_clustermap`.
- Old names are **removed** (no deprecation aliases). Update call sites to the new names.

### Fixed

- `analyze_niche_membership` previously called the clustermap function twice when `file_path` was provided.

### Preserved

- `adata.obsm['dist_e']`, `adata.uns['dist_e']`, `adata.obsm['dist_e_agg']`, and `adata.obs['niche_composition_cluster']` keys are unchanged, so existing h5ad artifacts and downstream figure scripts continue to work.

### Added

- `mievformer.__version__` attribute.

## Version 0.1.0 (2025)

Initial release of Mievformer.

### Features

- **Core Model Functions**
  - `optimize_nicheformer`: Train the Mievformer model with masked self-supervised learning
  - `calculate_wb_ez`: Calculate weight and bias terms for downstream analysis

- **Distribution Analysis** *(renamed in 0.2.0; original names shown)*
  - `calculate_spatial_distribution`: Compute spatial distribution of cells across microenvironments
  - `aggregate_dist_e`: Aggregate distribution embeddings

- **Downstream Analysis** *(renamed in 0.2.0; original names shown)*
  - `estimate_population_density`: Estimate cell population density in microenvironments
  - `analyze_density_correlation`: Analyze correlation between density and gene expression
  - `analyze_niche_composition`: Cluster and visualize niche composition

### Documentation

- Comprehensive API reference
- Getting started tutorial
- Methodology overview
