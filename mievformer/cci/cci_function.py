
import numpy as np
import pandas as pd
from scipy.stats import zscore
import anndata as ad
from scipy import sparse

def calculate_mean_expression(adata, dist, cell_type_col, layer='X'):
    cell_types = adata.obs[cell_type_col].unique()
    ref_indices = adata.uns['ref_adata_obs'].index
    mean_expression = {}
    for cell_type in cell_types:
        cell_indices = adata.uns['ref_adata_obs'][cell_type_col] == cell_type
        weighted_expression = dist[:, cell_indices].dot(adata[ref_indices].layers[layer][cell_indices, :])
        mean_expression[cell_type] = weighted_expression / dist[:, cell_indices].sum(axis=1).mean()
    return mean_expression

def cell_cell_interaction(adata, lr_df, cell_type_col, ligand_col='ligand', receptor_col='receptor', copy=False):
    if copy:
        adata = adata.copy()
    
    # Convert adata.X to array if it is a sparse matrix
    if isinstance(adata.X, sparse.spmatrix):
        adata.X = adata.X.toarray()
    
    # Standardize gene expression
    adata.layers['norm_exp'] = zscore(adata.X, axis=0)
    
    # Clip negative values
    adata.layers['norm_exp'] = np.clip(adata.layers['norm_exp'], 0, None)
    
    # Calculate mean expression for each cell type
    dist = np.exp(adata.obsm['dist'])
    mean_expression = calculate_mean_expression(adata, dist, cell_type_col, layer='norm_exp')
    
    # Save mean expression to adata.layers
    for cell_type, expr in mean_expression.items():
        adata.layers[f'mean_expression_{cell_type}'] = expr
    
    # Initialize results
    results = []
    
    # Calculate interaction scores
    for _, row in lr_df.iterrows():
        ligand, receptor = row[ligand_col], row[receptor_col]
        if ligand in adata.var_names and receptor in adata.var_names:
            for sender in mean_expression:
                for receiver in mean_expression:
                    ligand_expr = mean_expression[sender][adata.var_names.get_loc(ligand)]
                    receptor_expr = mean_expression[receiver][adata.var_names.get_loc(receptor)]
                    score = np.mean(ligand_expr * receptor_expr)
                    results.append([sender, receiver, ligand, receptor, score])
    
    # Create results DataFrame
    results_df = pd.DataFrame(results, columns=['sender', 'receiver', 'ligand', 'receptor', 'score'])
    
    return adata, results_df

import anndata
import pandas as pd
import argparse
from nicheformer.cci.cci_function import cell_cell_interaction

def main(input_path, lr_df_path, output_file, ligand_col, receptor_col, celltype_col='cell_type'):
    # Load the AnnData object
    adata = anndata.read_h5ad(input_path)

    # Load the ligand-receptor pairs
    lr_df = pd.read_csv(lr_df_path, sep='\t')

    # Perform cell-cell interaction analysis
    adata, results_df = cell_cell_interaction(
        adata,
        lr_df,
        cell_type_col=celltype_col,
        ligand_col=ligand_col,
        receptor_col=receptor_col
    )

    # Save the results
    results_df.to_csv(output_file, index=False)
    print(f"CCI estimation completed and results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CCI estimation.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input h5ad file.')
    parser.add_argument('--lr_df_path', type=str, required=True, help='Path to the ligand-receptor pairs file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output CSV file.')
    parser.add_argument('--ligand_col', type=str, default='ligand_gene_symbol', help='Column name for ligands in the LR pairs file.')
    parser.add_argument('--receptor_col', type=str, default='receptor_gene_symbol', help='Column name for receptors in the LR pairs file.')
    parser.add_argument('--celltype_col', type=str, default='cell_type', help='Column name for cell types in the AnnData object.')
    
    args = parser.parse_args()
    main(args.input_path, args.lr_df_path, args.output_file, args.ligand_col, args.receptor_col, args.celltype_col)
