"""High-level training and post-processing pipeline for the public API."""

from __future__ import annotations

import os

import anndata
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from . import nicheformer as nf
from . import utils
from . import workflow as wl


def _cell_representation(adata, key):
    if key == 'X':
        values = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    else:
        if key not in adata.obsm:
            raise KeyError(f"obsm[{key!r}] not found")
        values = adata.obsm[key]
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != adata.n_obs:
        raise ValueError('The cell representation must be a cell-by-feature matrix')
    if not np.isfinite(values).all():
        raise ValueError('The cell representation contains non-finite values')
    return values


def _batch_context(adata, batch_key):
    if batch_key == 'same':
        batch_key = None
    contract = adata.uns.get('mievformer_batch_contract', {})
    if batch_key is None and isinstance(contract, dict):
        contract_key = contract.get('batch_key')
        if contract_key in adata.obs:
            batch_key = str(contract_key)
    if batch_key is None:
        return None, (), np.repeat('', adata.n_obs)
    if batch_key not in adata.obs:
        raise KeyError(f"obs[{batch_key!r}] not found")
    series = pd.Series(adata.obs[batch_key], index=adata.obs_names)
    if series.isna().any():
        raise ValueError(f"obs[{batch_key!r}] contains missing sample labels")
    labels = series.astype(str).to_numpy()
    observed = set(labels)
    contract_order = []
    if isinstance(contract, dict) and contract.get('batch_key') == batch_key:
        contract_order = [str(value) for value in contract.get('batch_order', [])]
    if contract_order:
        if len(contract_order) != len(set(contract_order)):
            raise ValueError('Persisted Mievformer batch order contains duplicates')
        if set(contract_order) != observed:
            raise ValueError(
                'Persisted Mievformer batch order does not match observed samples'
            )
        order = tuple(contract_order)
    else:
        order = tuple(str(value) for value in pd.unique(labels))
    if not order:
        raise ValueError(f"obs[{batch_key!r}] does not contain any samples")
    return str(batch_key), order, labels


def _ordered_by_batch(adata, batch_key, order, labels):
    if batch_key is None:
        return adata.copy(), np.arange(adata.n_obs, dtype=np.int64)
    positions = np.concatenate(
        [np.flatnonzero(labels == sample) for sample in order]
    ).astype(np.int64)
    ordered = adata[positions].copy()
    ordered.obs[batch_key] = pd.Categorical(
        ordered.obs[batch_key].astype(str),
        categories=list(order),
        ordered=True,
    )
    ordered.uns['mievformer_batch_contract'] = {
        'batch_key': batch_key,
        'batch_order': list(order),
        'batch_counts': {
            sample: int(np.count_nonzero(labels == sample)) for sample in order
        },
        'batch_order_policy': 'first_observed_sample_order',
        'training_inference_mapping_equal': True,
        'ground_truth_used': False,
    }
    return ordered, positions


def _scale_spatial_coordinates(adata, batch_key, order, neighbor_num):
    if 'spatial' not in adata.obsm:
        raise KeyError("obsm['spatial'] not found")
    spatial = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    if spatial.ndim != 2 or spatial.shape[0] != adata.n_obs:
        raise ValueError("obsm['spatial'] must be a cell-by-coordinate matrix")
    if not np.isfinite(spatial).all():
        raise ValueError("obsm['spatial'] contains non-finite values")
    if int(neighbor_num) < 2:
        raise ValueError('neighbor_num must be at least 2')

    if batch_key is None:
        groups = [('single_slice', np.arange(adata.n_obs, dtype=np.int64))]
    else:
        labels = adata.obs[batch_key].astype(str).to_numpy()
        groups = [(sample, np.flatnonzero(labels == sample)) for sample in order]

    kth_distances = []
    counts = {}
    for sample, indices in groups:
        counts[str(sample)] = int(indices.size)
        if indices.size < int(neighbor_num):
            raise ValueError(
                f"Sample {sample!r} has {indices.size} cells, fewer than "
                f"neighbor_num={neighbor_num}"
            )
        distances = NearestNeighbors(n_neighbors=int(neighbor_num)).fit(
            spatial[indices]
        ).kneighbors(spatial[indices], return_distance=True)[0]
        kth_distances.append(distances[:, -1])
    scale = float(np.concatenate(kth_distances).mean())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('The spatial reference distance must be finite and positive')
    adata.obsm['spatial'] = (spatial / scale).astype(np.float32)
    adata.uns['mievformer_spatial_scale'] = {
        'scale': scale,
        'neighbor_num': int(neighbor_num),
        'sample_counts': counts,
        'policy': 'mean_slice_local_kth_neighbor_distance',
    }


def _split_one_group(group, neighbor_num):
    if group.n_obs < 4:
        raise ValueError('Each spatial sample needs at least four cells')
    validation = wl.clip_center_adata(group, 0.1)
    target = max(2, int(np.ceil(group.n_obs * 0.1)))
    target = min(target, group.n_obs - 2)
    if validation.n_obs < 2 or validation.n_obs >= group.n_obs - 1:
        spatial = np.asarray(group.obsm['spatial'])
        center = np.median(spatial, axis=0)
        selected = np.argsort(np.linalg.norm(spatial - center, axis=1))[:target]
        validation = group[np.sort(selected)].copy()
    training = group[~group.obs_names.isin(validation.obs_names)].copy()
    if training.n_obs < 2 or validation.n_obs < 2:
        raise ValueError('Training and validation partitions both need at least two cells')
    return training, validation


def _split_training_validation(adata, batch_key, order, neighbor_num):
    if batch_key is None:
        training, validation = _split_one_group(adata, neighbor_num)
        return training, validation
    training_blocks = []
    validation_blocks = []
    for sample in order:
        group = adata[adata.obs[batch_key].astype(str) == sample].copy()
        training, validation = _split_one_group(group, neighbor_num)
        training_blocks.append(training)
        validation_blocks.append(validation)
    concat_kwargs = {'axis': 0, 'merge': 'same', 'uns_merge': 'same'}
    return (
        anndata.concat(training_blocks, **concat_kwargs),
        anndata.concat(validation_blocks, **concat_kwargs),
    )


def _minimum_group_size(adata, batch_key):
    if batch_key is None:
        return int(adata.n_obs)
    return int(adata.obs[batch_key].astype(str).value_counts().min())


def _batch_one_hot(labels, order):
    mapping = {
        sample: vector
        for sample, vector in zip(order, np.eye(len(order), dtype=np.float32))
    }
    return np.stack([mapping[str(label)] for label in labels]).astype(np.float32)


def optimize(
    adata,
    model_path,
    *,
    ngpu,
    batch_size,
    max_epochs,
    neighbor_num,
    latent_dim,
    kld_ld,
    pent_ld,
    dist_space,
    cellrep_key,
    batch_key,
    batch_correct,
    representation_mode,
    ca_reference_num,
    ca_n_components,
    ca_reference_seed,
    ca_device,
    niche_n_neighbors,
    leiden_resolution,
    umap_min_dist,
    random_state,
):
    """Implement the public optimize_nicheformer pipeline."""
    mode = str(representation_mode).strip().lower()
    if mode not in {'auto', 'raw'}:
        raise ValueError("representation_mode must be 'auto' or 'raw'")
    if not adata.obs_names.is_unique:
        raise ValueError('adata.obs_names must be unique')
    result = adata.copy()
    original_obs_names = result.obs_names.copy()
    result.obsm['nf_cellrep'] = _cell_representation(result, cellrep_key)

    batch_key, batch_order, batch_labels = _batch_context(result, batch_key)
    resolved_batch_correct = wl.resolve_mievformer_batch_correction(
        result,
        batch_key=batch_key,
        batch_correct=batch_correct,
    )
    if len(batch_order) > 1 and mode == 'auto' and not resolved_batch_correct:
        raise ValueError(
            "Multi-slice representation_mode='auto' requires batch_correct='auto' "
            "or True; no ordinary-CA fallback is used"
        )

    result, _ = _ordered_by_batch(result, batch_key, batch_order, batch_labels)
    _scale_spatial_coordinates(
        result,
        batch_key=batch_key,
        order=batch_order,
        neighbor_num=int(neighbor_num),
    )
    train_adata, val_adata = _split_training_validation(
        result,
        batch_key=batch_key,
        order=batch_order,
        neighbor_num=int(neighbor_num),
    )
    train_neighbor_num = min(int(neighbor_num), _minimum_group_size(train_adata, batch_key))
    val_neighbor_num = min(int(neighbor_num), _minimum_group_size(val_adata, batch_key))
    ds_train = wl.adata2ds(
        train_adata,
        neighbor_num=train_neighbor_num,
        batch_key=batch_key,
    )
    ds_val = wl.adata2ds(
        val_adata,
        neighbor_num=val_neighbor_num,
        batch_key=batch_key,
    )

    log_dir = os.path.dirname(os.fspath(model_path))
    model_stem = os.path.splitext(os.path.basename(os.fspath(model_path)))[0]
    checkpoint_name = f'{model_stem}_checkpoints'
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(log_dir, checkpoint_name)
        )
        logger = CSVLogger(save_dir=log_dir, version=1, name='lightning_logs')
    else:
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_name)
        logger = CSVLogger(save_dir='.', version=1, name='lightning_logs')
    accelerator = 'gpu' if torch.cuda.is_available() and int(ngpu) > 0 else 'cpu'
    devices = int(ngpu) if accelerator == 'gpu' else 1
    trainer = pl.Trainer(
        max_epochs=int(max_epochs),
        devices=devices,
        accelerator=accelerator,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=20),
            checkpoint_callback,
        ],
        reload_dataloaders_every_n_epochs=1,
        strategy=(
            'ddp_find_unused_parameters_true'
            if accelerator == 'gpu' and int(ngpu) > 1
            else 'auto'
        ),
        logger=logger,
    )
    epoch_size = min(100000, max(len(ds_train), int(batch_size)))
    model_kwargs = {
        'input_dim': result.obsm['nf_cellrep'].shape[1],
        'latent_dim': int(latent_dim),
        'train_ds': ds_train,
        'val_ds': ds_val,
        'kld_ld': float(kld_ld),
        'pent_ld': float(pent_ld),
        'dist_space': dist_space,
        'batch_size': int(batch_size),
        'batch_correct': resolved_batch_correct,
        'epoch_size': epoch_size,
    }
    model = nf.NicheFormer(**model_kwargs)
    val_loader = DataLoader(ds_val, batch_size=int(batch_size), num_workers=0)
    trainer.fit(model, val_dataloaders=val_loader)
    model = nf.NicheFormer.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        **model_kwargs,
    )
    torch.save(model.state_dict(), model_path)

    ds_full = wl.adata2ds(result, neighbor_num=int(neighbor_num), batch_key=batch_key)
    raw_e = utils.output_niche_rep(ds_full, model, batch_size=int(batch_size)).numpy()
    result.obsm['e'] = np.asarray(raw_e, dtype=np.float32)
    if resolved_batch_correct:
        ordered_labels = result.obs[batch_key].astype(str).to_numpy()
        result.obsm['batch_one_hot'] = _batch_one_hot(ordered_labels, batch_order)
    wl.add_wb_ez(result, model, cell_rep_key='nf_cellrep')

    result = result[original_obs_names].copy()
    if not result.obs_names.equals(original_obs_names):
        raise AssertionError('Failed to restore the original cell order')

    if mode == 'auto':
        ca_kwargs = {
            'reference_seed': int(ca_reference_seed),
            'n_components': ca_n_components,
            'device': ca_device,
        }
        if ca_reference_num is not None:
            ca_kwargs['reference_num'] = int(ca_reference_num)
        result, _ = wl.add_mievformer_ca_features(
            result,
            model=model,
            sample_key=batch_key,
            strategy='auto',
            feature_key=wl.MIEVFORMER_CA_KEY,
            cell_rep_key='nf_cellrep',
            set_as_default=True,
            **ca_kwargs,
        )
    else:
        result.uns[wl.MIEVFORMER_DEFAULT_REPRESENTATION_KEY] = {
            'default_embedding_key': 'e',
            'raw_embedding_key': 'e',
            'strategy': 'raw_e_explicit',
            'sample_key': '' if batch_key is None else batch_key,
            'n_samples': max(1, len(batch_order)),
            'selection_policy': 'explicit_raw_benchmark_override',
            'raw_embedding_retained': True,
            'silent_fallback_used': False,
        }

    graph_neighbors = min(int(niche_n_neighbors), result.n_obs - 1)
    if graph_neighbors < 2:
        raise ValueError('At least three cells are required for niche clustering')
    sc.pp.neighbors(
        result,
        use_rep='e',
        n_neighbors=graph_neighbors,
        random_state=int(random_state),
    )
    sc.tl.umap(
        result,
        min_dist=float(umap_min_dist),
        random_state=int(random_state),
    )
    sc.tl.leiden(
        result,
        key_added='leiden_e',
        resolution=float(leiden_resolution),
        random_state=int(random_state),
    )
    result.uns['mievformer_niche_graph'] = {
        'use_rep': 'e',
        'representation_strategy': result.uns[
            wl.MIEVFORMER_DEFAULT_REPRESENTATION_KEY
        ]['strategy'],
        'n_neighbors': graph_neighbors,
        'leiden_resolution': float(leiden_resolution),
        'umap_min_dist': float(umap_min_dist),
        'random_state': int(random_state),
    }
    return result


def calculate_weights(
    adata,
    model_path,
    *,
    batch_key,
    batch_correct,
    neighbor_num,
    latent_dim,
    cellrep_key,
):
    """Calculate score-function weights while respecting the raw-e contract."""
    adata.obsm['nf_cellrep'] = _cell_representation(adata, cellrep_key)
    batch_key, order, labels = _batch_context(adata, batch_key)
    resolved_batch_correct = wl.resolve_mievformer_batch_correction(
        adata,
        batch_key=batch_key,
        batch_correct=batch_correct,
    )
    model = wl.loading_pre_trained_model(
        model_path,
        adata,
        {
            'ldim': int(latent_dim),
            'crkey': 'nf_cellrep',
            'bkey': batch_key,
            'bcorr': resolved_batch_correct,
        },
    )

    metadata = adata.uns.get(wl.MIEVFORMER_DEFAULT_REPRESENTATION_KEY, {})
    raw_key = metadata.get('raw_embedding_key') if isinstance(metadata, dict) else None
    has_raw = raw_key in adata.obsm if raw_key else 'e' in adata.obsm
    if not has_raw:
        ordered, positions = _ordered_by_batch(adata, batch_key, order, labels)
        ds = wl.adata2ds(ordered, neighbor_num=int(neighbor_num), batch_key=batch_key)
        ordered_raw = utils.output_niche_rep(ds, model).numpy()
        raw = np.empty_like(ordered_raw)
        raw[positions] = ordered_raw
        if 'e' in adata.obsm:
            adata.obsm[wl.MIEVFORMER_RAW_E_KEY] = raw
            if isinstance(metadata, dict):
                metadata['raw_embedding_key'] = wl.MIEVFORMER_RAW_E_KEY
        else:
            adata.obsm['e'] = raw
    if resolved_batch_correct:
        adata.obsm['batch_one_hot'] = _batch_one_hot(labels, order)
    wl.add_wb_ez(adata, model, cell_rep_key='nf_cellrep')
    return adata
