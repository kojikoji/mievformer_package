import numpy as np
from scipy import stats
import jax.numpy as jnp
import jax
from jax import lax, nn
from jax.scipy.stats import multivariate_normal
import optax
import functools
import anndata
import scanpy as sc



# @functools.partial(jax.vmap, in_axes=(0, None, None))
@jax.jit
def gauss_loss(params: optax.Params, x: jnp.ndarray, L: jnp.ndarray) -> jnp.ndarray:
    gene_num = x.shape[1]
    a, b = nn.softplus(params[:gene_num]), params[gene_num:]
    a = a / jnp.linalg.norm(a)
    trans_x = x * a + b
    trans_x  = trans_x 
    y = lax.linalg.triangular_solve(L, trans_x, lower=True, transpose_a=True)
    nlprob = (y * y).sum()
    return nlprob

class IncGene:
    def __init__(self, ref_adata, kernel_mode='linear'):
        self.ref_adata = ref_adata.copy()
        sc.pp.scale(self.ref_adata)
        self.kernel_mode = kernel_mode
        self.cov_eps = 1.0e-6
    
    def fit(self, target_adata, iter_num=100):
        self.input_genes = np.intersect1d(target_adata.var_names, self.ref_adata.var_names)
        self.predicted_genes = self.ref_adata.var_names[~np.isin(self.ref_adata.var_names, self.input_genes)]
        self.all_genes = np.concatenate([self.input_genes, self.predicted_genes])
        self.ref_adata = self.ref_adata[:, self.all_genes]
        target_adata = target_adata.copy()
        sc.pp.scale(target_adata)
        input_X = jnp.array(target_adata[:, self.input_genes].X)
        ref_X = jnp.array(self.ref_adata.X)
        if self.kernel_mode == 'linear':
            self.cov = ref_X.T @ ref_X
            self.input_cov = self.cov[:len(self.input_genes), :len(self.input_genes)]
            for i in range(6):
                try:
                    self.L = lax.linalg.cholesky(self.input_cov + self.cov_eps * jnp.eye(self.input_cov.shape[0]))
                    assert jnp.all(jnp.isfinite(self.L))
                    break
                except:
                    self.eps_cov *= 10
                    print(f'Error: Cholesky decomposition failed. Retry with cov_eps = {self.cov_eps}')
        params = jnp.concatenate([jnp.zeros(self.input_genes.shape[0]), jnp.zeros(self.input_genes.shape[0])])
        optimizer = optax.adam(1e-2)
        opt_state = optimizer.init(params)
        for _ in range(iter_num):
            loss, grads = jax.value_and_grad(gauss_loss)(params, input_X, self.L)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            if _ % 100 == 0:
                print(f'Iter {_}, Loss: {loss}')
        gene_num = len(self.input_genes)
        self.a = nn.softplus(params[:gene_num])
        self.a = self.a / jnp.linalg.norm(self.a)
        self.b = params[gene_num:]
        self.pr_in_cov = self.cov[len(self.input_genes):, :len(self.input_genes)]
        self.pr_in_cov_lt_inv = lax.linalg.triangular_solve(self.L, self.pr_in_cov, lower=True, transpose_a=True)

    def predict(self, target_adata):
        target_adata = target_adata.copy()
        sc.pp.scale(target_adata)
        input_X = jnp.array(target_adata[:, self.input_genes].X)
        trans_X = input_X * self.a + self.b
        trans_X  = trans_X / jnp.linalg.norm(trans_X, axis=1, keepdims=True)
        y = lax.linalg.triangular_solve(self.L, trans_X, lower=True, transpose_a=True)
        mu = y @ self.pr_in_cov_lt_inv.T
        all_X = np.array(jnp.concatenate([trans_X, mu], axis=1))
        pred_target_adata = anndata.AnnData(X=all_X, obs=target_adata.obs, var=self.ref_adata.var)
        return pred_target_adata




        