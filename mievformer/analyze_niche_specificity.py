import torch
import math
import torch.utils
import scanpy as sc


@torch.no_grad()
def calc_lk(model, adata, ref_adata, cr_key='X_pca', latent_key='e', device='cuda'):
    ref_e = torch.tensor(ref_adata.obsm[latent_key], dtype=torch.float32)
    z = torch.tensor(adata.obsm[cr_key], dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(z)
    model.eval()
    model.to(device)
    ref_e = ref_e.to(device)
    lk_list = []
    data_loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    for z in data_loader:
        w_e = model.distributor.e2w(ref_e)
        wb_z = model.distributor.z2wb(z[0].to(device))
        w_z, b_z = wb_z[..., :-1], wb_z[..., -1]
        lk = w_e @ w_z.T + b_z
        lk_list.append(lk.detach().cpu().T)
    lk = torch.cat(lk_list, dim=0)
    return lk
    



@torch.no_grad()
def calc_niche_specificity(model, adata, ref_num=1000, cr_key='X_pca', latent_key='e', device='cuda', celltype_key=None):
    ref_adata = sc.pp.subsample(adata, n_obs=ref_num, copy=True)
    lk = calc_lk(model, adata, ref_adata, cr_key, latent_key, device)
    if celltype_key is not None:
        celltypes = adata.obs[celltype_key].values
        celltype_lk = {
            celltype: torch.logsumexp(lk[celltypes == celltype], dim=0, keepdim=True)
            for celltype in set(celltypes)
        }
        ct_lk_mat = torch.cat([
            celltype_lk[celltype]
            for celltype in celltypes
        ], dim=0)
        lk = lk - ct_lk_mat
    else:
        lk = lk - torch.logsumexp(lk, dim=0, keepdim=True)
    leb = lk - torch.logsumexp(lk, dim=1, keepdim=True)
    niche_specifity = ((leb + math.log(ref_num)) * torch.exp(leb)).sum(dim=1)
    return niche_specifity.numpy()
    