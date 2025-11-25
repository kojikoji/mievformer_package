import unittest
import torch
import pandas as pd
import numpy as np
from anndata import AnnData
from .binary_cci import calculate_coactivate_probs, apply_binary_cci_to_adata, BinaryCCI

class TestBinaryCCI(unittest.TestCase):
    def setUp(self):
        # Create a mock AnnData object
        self.adata = AnnData(
            X=np.random.randint(0, 50, size=(100, 50)),
            obs=pd.DataFrame({
                'celltype': pd.Categorical(['type1', 'type2', 'type3', 'type4'] * 25)
            }),
            var=pd.DataFrame(index=[f'gene{i}' for i in range(50)]),
            obsm={'e': np.random.rand(100, 10)}
        )
        self.adata.layers['counts'] = self.adata.X

        # Create a mock ligand-receptor DataFrame
        self.lt_df = pd.DataFrame({
            'ligand': ['gene1', 'gene2', 'gene3'],
            'receptor': ['gene4', 'gene5', 'gene6']
        })

        # Create a mock model
        self.model = BinaryCCI(batch_size=32, celltype_dim=4, phi_dim=1028, hidden_dim=256, gene_dim=50, e_dim=10)

    def test_calculate_coactivate_probs(self):
        coactivate_prob_df, adata = calculate_coactivate_probs(
            self.adata, self.model, self.lt_df, ligand_col='ligand', receptor_col='receptor', e_feature_label='e', celltype_label='celltype'
        )

        # Check the DataFrame columns
        expected_columns = ['sender', 'receiver', 'ligand', 'receptor', 'coactivate_prob']
        self.assertTrue(all(col in coactivate_prob_df.columns for col in expected_columns))

        # Check the DataFrame content
        self.assertEqual(len(coactivate_prob_df), 4 * 4 * 3)
        self.assertTrue(all(coactivate_prob_df['ligand'].isin(self.lt_df['ligand'])))
        self.assertTrue(all(coactivate_prob_df['receptor'].isin(self.lt_df['receptor'])))

    def test_apply_binary_cci_to_adata(self):
        model, adata = apply_binary_cci_to_adata(self.adata, max_epochs=1, batch_size=32, phi_dim=1028, celltype_label='celltype', e_feature_label='e')

        # Check if the model is an instance of BinaryCCI
        self.assertIsInstance(model, BinaryCCI)

        # Check if the adata object is returned correctly
        self.assertIsInstance(adata, AnnData)
        self.assertTrue((adata.X == self.adata.X).all())
        self.assertTrue((adata.obs['celltype'] == self.adata.obs['celltype']).all())
        self.assertTrue((adata.obsm['e'] == self.adata.obsm['e']).all())

if __name__ == '__main__':
    unittest.main()