from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import softmax

import mievformer as mf
from mievformer import workflow as wl


def make_weighted_adata(n_obs=48, latent_dim=5, seed=3):
    rng = np.random.default_rng(seed)
    adata = ad.AnnData(np.zeros((n_obs, 1), dtype=np.float32))
    adata.obsm['w_e'] = rng.normal(size=(n_obs, latent_dim)).astype(np.float32)
    adata.obsm['w_z'] = rng.normal(size=(n_obs, latent_dim)).astype(np.float32)
    adata.obsm['b_z'] = rng.normal(size=n_obs).astype(np.float32)
    adata.obsm['e'] = rng.normal(size=(n_obs, 3)).astype(np.float32)
    return adata


def test_reference_probabilities_match_direct_softmax():
    adata = make_weighted_adata(n_obs=24)
    reference_indices = np.array([1, 4, 7, 13, 18, 22])
    observed = wl.calculate_reference_probabilities(
        adata.obsm['w_e'],
        adata.obsm['w_z'][reference_indices],
        adata.obsm['b_z'][reference_indices],
        batch_size=7,
        device='cpu',
    )
    logits = (
        adata.obsm['w_e'] @ adata.obsm['w_z'][reference_indices].T
        + np.asarray(adata.obsm['b_z'][reference_indices]).reshape(-1)
    )
    np.testing.assert_allclose(observed, softmax(logits, axis=1), atol=2e-7)
    np.testing.assert_allclose(observed.sum(axis=1), 1.0, atol=2e-6)


def test_single_slice_ca_is_default_and_roundtrips(tmp_path):
    adata = make_weighted_adata()
    raw_e = adata.obsm['e'].copy()
    result, model = wl.add_mievformer_ca_features(
        adata,
        reference_num=16,
        reference_seed=2,
        n_components=4,
        max_fit_cells=36,
        fit_seed=5,
        device='cpu',
        copy=True,
    )
    assert model['strategy'] == wl.MIEVFORMER_SINGLE_SLICE_CA
    np.testing.assert_array_equal(result.obsm[wl.MIEVFORMER_RAW_E_KEY], raw_e)
    np.testing.assert_array_equal(result.obsm['e'], result.obsm[wl.MIEVFORMER_CA_KEY])
    output = tmp_path / 'single.h5ad'
    result.write_h5ad(output)
    restored = ad.read_h5ad(output)
    np.testing.assert_array_equal(restored.obsm['e'], result.obsm['e'])


def make_multi_slice_adata(n_obs=40, seed=71):
    rng = np.random.default_rng(seed)
    labels = np.repeat(['slice_b', 'slice_a'], n_obs // 2)
    one_hot = np.eye(2, dtype=np.float32)[np.repeat([0, 1], n_obs // 2)]
    adata = ad.AnnData(np.zeros((n_obs, 1), dtype=np.float32))
    adata.obs_names = [f'cell_{index:03d}' for index in range(n_obs)]
    adata.obs['slice'] = pd.Categorical(
        labels, categories=['slice_b', 'slice_a'], ordered=True
    )
    adata.obsm['e'] = rng.normal(size=(n_obs, 3)).astype(np.float32)
    adata.obsm['w_z'] = rng.normal(size=(n_obs, 4)).astype(np.float32)
    adata.obsm['b_z'] = rng.normal(size=n_obs).astype(np.float32)
    adata.obsm['batch_one_hot'] = one_hot
    torch.manual_seed(seed)
    e2w = torch.nn.Linear(5, 4)
    e2w.eval()
    with torch.no_grad():
        conditional = np.concatenate([adata.obsm['e'], one_hot], axis=1)
        adata.obsm['w_e'] = e2w(torch.from_numpy(conditional)).numpy()
    model = SimpleNamespace(distributor=SimpleNamespace(e2w=e2w))
    return adata, model


def test_multi_slice_sample_conditional_ca_is_balanced_and_roundtrips(tmp_path):
    adata, model = make_multi_slice_adata()
    result, ca_model = wl.add_mievformer_ca_features(
        adata,
        model=model,
        sample_key='slice',
        reference_num=10,
        reference_seed=3,
        n_components=3,
        max_fit_cells=30,
        device='cpu',
        copy=True,
    )
    assert ca_model['strategy'] == wl.MIEVFORMER_MULTI_SLICE_CA
    assert ca_model['distributor_reproduction']['passed'] is True
    assert ca_model['fit_probability_audits']['reference_seed_3']['quotas'] == {
        'slice_b': 5,
        'slice_a': 5,
    }
    assert np.isfinite(result.obsm['e']).all()
    output = tmp_path / 'multi.h5ad'
    result.write_h5ad(output)
    restored = ad.read_h5ad(output)
    projected = wl.transform_mievformer_ca(
        restored,
        restored.uns[wl.MIEVFORMER_CA_KEY],
        model=model,
        device='cpu',
    )
    np.testing.assert_allclose(projected, restored.obsm['e'], atol=2e-6)


def test_multi_slice_missing_conditioning_never_falls_back():
    adata, model = make_multi_slice_adata()
    del adata.obsm['batch_one_hot']
    with pytest.raises(ValueError, match='ordinary CA is not used as a fallback'):
        wl.add_mievformer_ca_features(
            adata,
            model=model,
            sample_key='slice',
            reference_num=10,
            n_components=3,
            device='cpu',
        )


def test_auto_policy_uses_slice_count():
    single = make_weighted_adata()
    multi, _ = make_multi_slice_adata()
    assert wl.resolve_mievformer_ca_strategy(single) == wl.MIEVFORMER_SINGLE_SLICE_CA
    assert (
        wl.resolve_mievformer_ca_strategy(multi, sample_key='slice')
        == wl.MIEVFORMER_MULTI_SLICE_CA
    )
    assert wl.resolve_mievformer_batch_correction(single) is False
    assert wl.resolve_mievformer_batch_correction(
        multi, batch_key='slice'
    ) is True


def test_auto_dimension_uses_mean_relative_eigengap_grid():
    spectrum = np.geomspace(100.0, 10.0, num=40)
    spectrum[9:] *= 0.1
    spectra = np.vstack([spectrum * (1 + seed * 0.001) for seed in range(10)])
    selected, diagnostics = wl.select_reference_probability_ca_dimension(
        spectra,
        candidate_dimensions=(5, 10, 20, 40),
        expected_n_reference_spectra=10,
    )
    assert selected == 10
    assert diagnostics['raw_selected_n_components'] == 9
    assert diagnostics['ground_truth_used'] is False


def test_nonfinite_ca_inputs_are_rejected():
    adata = make_weighted_adata()
    adata.obsm['w_e'][0, 0] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        wl.fit_reference_probability_ca(
            adata,
            reference_num=12,
            n_components=3,
            device='cpu',
        )


def test_standard_multibatch_rejects_disabled_batch_correction(tmp_path):
    adata, _ = make_multi_slice_adata()
    rng = np.random.default_rng(8)
    adata.obsm['X_pca'] = rng.normal(size=(adata.n_obs, 5)).astype(np.float32)
    adata.obsm['spatial'] = rng.normal(size=(adata.n_obs, 2)).astype(np.float32)
    with pytest.raises(ValueError, match='requires batch_correct'):
        mf.optimize_nicheformer(
            adata,
            tmp_path / 'unused.pth',
            batch_key='slice',
            batch_correct=False,
        )
