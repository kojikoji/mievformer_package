
# Cell-Cell Interaction Analysis

This module provides a function for analyzing cell-cell interactions based on ligand-receptor pairs.

## Function: `cell_cell_interaction`

### Parameters:
- `adata` (anndata.AnnData): The annotated data matrix.
- `lr_df` (pandas.DataFrame): DataFrame containing ligand-receptor pairs with columns for ligand and receptor.
- `cell_type_col` (str): The column name in `adata.obs` that contains cell type information.
- `ligand_col` (str, optional): The column name in `lr_df` that contains ligand information. Default is `ligand`.
- `receptor_col` (str, optional): The column name in `lr_df` that contains receptor information. Default is `receptor`.
- `copy` (bool, optional): Whether to return a copy of the AnnData object. Default is `False`.

### Returns:
- `adata` (anndata.AnnData): The AnnData object with mean expression added to `obsm`.
- `results_df` (pandas.DataFrame): DataFrame containing interaction scores and p-values.

### Usage:
from nicheformer.cci.cci_function import cell_cell_interaction

# Example usage
adata, results_df = cell_cell_interaction(adata, lr_df, 'cell_type')

### Local Information:
- The file 'human_lr_pair.txt' located in 'db/lr' should be used for ligand-receptor data.
- Use 'ligand_gene_symbol' as the 'ligand_col' and 'receptor_gene_symbol' as the 'receptor_col'.
- Conda Environment: nicheformer
- PYTHONPATH: /home/tiisaishima/proj/nichedynamics
