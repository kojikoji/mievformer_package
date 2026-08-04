"""Sample-balanced, sample-conditional reference-probability CA.

This module contains the reusable numerical implementation used by the
Mievformer workflow.  For every reference sample, the batch-corrected
distributor is evaluated counterfactually with that sample's one-hot vector.
References are balanced across samples and every sample block receives equal
probability mass before correspondence analysis (CA).

The high-level AnnData API and the single- versus multi-slice default policy
live in :mod:`mievformer.workflow`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, Union

import numpy as np
import torch

from .nicheformer import Distributor


VARIANT = "sample_conditional_ca"


@dataclass(frozen=True)
class ReferenceLayout:
    """Balanced reference-cell layout, grouped into contiguous sample blocks."""

    indices: np.ndarray
    reference_samples: np.ndarray
    sample_order: tuple
    quotas: dict
    block_slices: tuple
    seed: int


def derive_sample_one_hot_mapping(
    batch_values: Sequence[Any],
    batch_one_hot: Any,
    *,
    expected_order: Sequence[Any],
    atol: float = 1.0e-7,
) -> dict:
    """Validate the persisted sample-to-one-hot mapping without re-encoding it."""
    batches = np.asarray(batch_values).astype(str)
    one_hot = np.asarray(batch_one_hot, dtype=np.float32)
    order = tuple(str(value) for value in expected_order)
    if batches.ndim != 1:
        raise ValueError("batch_values must be one-dimensional")
    if one_hot.ndim != 2 or one_hot.shape[0] != batches.size:
        raise ValueError("batch_one_hot must have one row per cell")
    if len(order) != len(set(order)) or set(order) != set(batches):
        raise ValueError("Expected sample order does not match observed samples")
    if one_hot.shape[1] != len(order):
        raise ValueError(
            f"One-hot width {one_hot.shape[1]} differs from sample count {len(order)}"
        )
    if not np.isfinite(one_hot).all():
        raise ValueError("batch_one_hot contains non-finite values")

    mapping = {}
    identity = np.eye(len(order), dtype=np.float32)
    for expected_column, sample in enumerate(order):
        rows = one_hot[batches == sample]
        if rows.shape[0] == 0:
            raise AssertionError(f"Observed sample has no rows: {sample}")
        vector = np.asarray(rows[0], dtype=np.float32)
        if float(np.max(np.abs(rows - vector[None, :]))) > atol:
            raise ValueError(f"Sample {sample!r} has non-constant one-hot rows")
        expected = identity[expected_column]
        if not np.allclose(vector, expected, atol=atol, rtol=0.0):
            raise ValueError(
                f"Sample {sample!r} one-hot vector differs from the persisted "
                f"training mapping: {vector.tolist()} != {expected.tolist()}"
            )
        mapping[sample] = vector.copy()
    return mapping


def equal_sample_quotas(
    sample_order: Sequence[Any],
    reference_num: int,
) -> dict:
    """Allocate references as evenly as possible in the persisted sample order."""
    order = tuple(str(value) for value in sample_order)
    if not order or len(order) != len(set(order)):
        raise ValueError("sample_order must be non-empty and unique")
    reference_num = int(reference_num)
    if reference_num < len(order):
        raise ValueError("reference_num must be at least the sample count")
    base, remainder = divmod(reference_num, len(order))
    quotas = {
        sample: base + int(index < remainder)
        for index, sample in enumerate(order)
    }
    if sum(quotas.values()) != reference_num:
        raise AssertionError("Reference quotas do not sum to reference_num")
    if max(quotas.values()) - min(quotas.values()) > 1:
        raise AssertionError("Reference quotas differ by more than one")
    return quotas


def select_balanced_reference_indices(
    batch_values: Sequence[Any],
    reference_num: int,
    seed: int,
    *,
    sample_order: Sequence[Any],
) -> ReferenceLayout:
    """Sample references without replacement and group them by sample."""
    batches = np.asarray(batch_values).astype(str)
    order = tuple(str(value) for value in sample_order)
    if batches.ndim != 1:
        raise ValueError("batch_values must be one-dimensional")
    if len(order) != len(set(order)) or set(order) != set(batches):
        raise ValueError("sample_order must contain each observed sample once")
    quotas = equal_sample_quotas(order, int(reference_num))
    rng = np.random.default_rng(int(seed))
    index_blocks = []
    sample_blocks = []
    block_slices = []
    start = 0
    for sample in order:
        available = np.flatnonzero(batches == sample)
        quota = quotas[sample]
        if available.size < quota:
            raise ValueError(
                f"Sample {sample!r} has {available.size} cells, below quota {quota}"
            )
        selected = np.sort(
            rng.choice(available, size=quota, replace=False)
        ).astype(np.int64)
        stop = start + quota
        index_blocks.append(selected)
        sample_blocks.append(np.repeat(sample, quota))
        block_slices.append((sample, start, stop))
        start = stop
    indices = np.concatenate(index_blocks)
    reference_samples = np.concatenate(sample_blocks).astype(str)
    if np.unique(indices).size != int(reference_num):
        raise AssertionError("Balanced reference selection contains duplicates")
    if not np.array_equal(batches[indices], reference_samples):
        raise AssertionError("Reference sample labels differ from selected cells")
    return ReferenceLayout(
        indices=indices,
        reference_samples=reference_samples,
        sample_order=order,
        quotas=quotas,
        block_slices=tuple(block_slices),
        seed=int(seed),
    )


def load_matching_distributor(
    checkpoint_path: Union[str, Path],
    *,
    z_input_dim: int,
    e_plus_batch_dim: int,
    device: Union[str, torch.device],
) -> Distributor:
    """Load only the distributor from a saved Mievformer state dictionary."""
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, Mapping):
        raise TypeError("Expected a Mievformer state-dict mapping")
    distributor_state = {
        str(key).removeprefix("distributor."): value
        for key, value in state.items()
        if str(key).startswith("distributor.")
    }
    if not any(key.startswith("e2w.") for key in distributor_state):
        raise KeyError("Mievformer state dict lacks distributor.e2w")
    distributor = Distributor(
        input_dim=int(z_input_dim),
        latent_dim=int(e_plus_batch_dim),
    )
    incompatible = distributor.load_state_dict(distributor_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Distributor state mismatch: {incompatible}")
    distributor.eval()
    distributor.requires_grad_(False)
    return distributor.to(torch.device(device))


def verify_observed_w_e(
    e: Any,
    batch_one_hot: Any,
    observed_w_e: Any,
    e2w: torch.nn.Module,
    *,
    batch_size: int,
    device: Union[str, torch.device],
    atol: float = 2.0e-6,
    rtol: float = 1.0e-5,
) -> dict:
    """Verify that the distributor and persisted one-hot rows reproduce ``w_e``."""
    e_values = np.asarray(e, dtype=np.float32)
    one_hot = np.asarray(batch_one_hot, dtype=np.float32)
    expected = np.asarray(observed_w_e, dtype=np.float32)
    if e_values.shape[0] != one_hot.shape[0] or expected.shape[0] != e_values.shape[0]:
        raise ValueError("Observed Mievformer arrays have inconsistent row counts")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for sample-conditional CA")
    e2w.to(resolved_device)
    e2w.eval()
    max_error = 0.0
    sum_error = 0.0
    element_count = 0
    with torch.no_grad():
        for start in range(0, e_values.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), e_values.shape[0])
            conditional = np.concatenate(
                [e_values[start:stop], one_hot[start:stop]],
                axis=1,
            )
            observed = (
                e2w(
                    torch.as_tensor(
                        conditional,
                        dtype=torch.float32,
                        device=resolved_device,
                    )
                )
                .cpu()
                .numpy()
            )
            target = expected[start:stop]
            difference = np.abs(observed - target)
            max_error = max(max_error, float(difference.max()))
            sum_error += float(difference.sum(dtype=np.float64))
            element_count += int(difference.size)
            if not np.allclose(observed, target, atol=atol, rtol=rtol):
                raise AssertionError(
                    "The sample-conditional distributor does not reproduce stored "
                    f"w_e for rows {start}:{stop}; max error="
                    f"{float(difference.max())}"
                )
    return {
        "passed": True,
        "n_cells": int(e_values.shape[0]),
        "atol": float(atol),
        "rtol": float(rtol),
        "max_absolute_error": max_error,
        "mean_absolute_error": sum_error / float(element_count),
    }


def _new_probability_audit(n_samples: int) -> dict:
    return {
        "variant": VARIANT,
        "n_batches": 0,
        "n_rows": 0,
        "all_finite": True,
        "max_row_sum_absolute_error": 0.0,
        "max_block_sum_absolute_error": 0.0,
        "block_target": 1.0 / float(n_samples),
    }


def iter_probability_batches(
    query_e: Any,
    reference_w_z: Any,
    reference_b_z: Any,
    layout: ReferenceLayout,
    *,
    e2w: torch.nn.Module,
    sample_one_hot: Mapping[str, np.ndarray],
    batch_size: int,
    device: Union[str, torch.device],
    audit: dict,
) -> Iterator[tuple]:
    """Yield probabilities with equal mass in every reference-sample block."""
    query = np.asarray(query_e, dtype=np.float32)
    w_z = np.asarray(reference_w_z, dtype=np.float32)
    b_z = np.asarray(reference_b_z, dtype=np.float32).reshape(-1)
    if query.ndim != 2 or w_z.ndim != 2 or b_z.shape[0] != w_z.shape[0]:
        raise ValueError("Invalid sample-conditional query/reference arrays")
    if not np.isfinite(query).all() or not np.isfinite(w_z).all() or not np.isfinite(b_z).all():
        raise ValueError("Sample-conditional query/reference arrays must be finite")
    if w_z.shape[0] != layout.indices.size:
        raise ValueError("Reference arrays do not match the balanced layout")
    if set(sample_one_hot) != set(layout.sample_order):
        raise ValueError("Sample one-hot mapping does not cover reference blocks")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for sample-conditional CA")
    e2w.to(resolved_device)
    e2w.eval()
    w_z_device = torch.as_tensor(w_z, dtype=torch.float32, device=resolved_device)
    b_z_device = torch.as_tensor(b_z, dtype=torch.float32, device=resolved_device)
    block_target = 1.0 / float(len(layout.sample_order))

    with torch.no_grad():
        for start in range(0, query.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), query.shape[0])
            query_device = torch.as_tensor(
                np.ascontiguousarray(query[start:stop]),
                dtype=torch.float32,
                device=resolved_device,
            )
            probabilities_device = torch.empty(
                (stop - start, w_z.shape[0]),
                dtype=torch.float32,
                device=resolved_device,
            )
            for sample, block_start, block_stop in layout.block_slices:
                vector_values = np.asarray(sample_one_hot[sample], dtype=np.float32)
                if vector_values.ndim != 1 or not np.isfinite(vector_values).all():
                    raise ValueError(f"Invalid one-hot vector for sample {sample!r}")
                vector = torch.as_tensor(
                    vector_values,
                    dtype=torch.float32,
                    device=resolved_device,
                ).expand(stop - start, -1)
                conditional_w_e = e2w(torch.cat([query_device, vector], dim=1))
                logits = (
                    conditional_w_e @ w_z_device[block_start:block_stop].T
                    + b_z_device[block_start:block_stop][None, :]
                )
                probabilities_device[:, block_start:block_stop] = (
                    torch.softmax(logits, dim=1) * block_target
                )
            probabilities = probabilities_device.cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError("Sample-conditional probabilities are non-finite")
            row_error = float(
                np.max(
                    np.abs(
                        probabilities.sum(axis=1, dtype=np.float64) - 1.0
                    )
                )
            )
            block_error = 0.0
            for _, block_start, block_stop in layout.block_slices:
                block_error = max(
                    block_error,
                    float(
                        np.max(
                            np.abs(
                                probabilities[:, block_start:block_stop].sum(
                                    axis=1,
                                    dtype=np.float64,
                                )
                                - block_target
                            )
                        )
                    ),
                )
            if row_error > 1.0e-5 or block_error > 1.0e-5:
                raise RuntimeError(
                    "Sample-conditional probability mass check failed: "
                    f"row={row_error}, block={block_error}"
                )
            audit["n_batches"] += 1
            audit["n_rows"] += stop - start
            audit["max_row_sum_absolute_error"] = max(
                audit["max_row_sum_absolute_error"],
                row_error,
            )
            audit["max_block_sum_absolute_error"] = max(
                audit["max_block_sum_absolute_error"],
                block_error,
            )
            yield start, stop, probabilities


def materialize_probabilities(
    query_e: Any,
    reference_w_z: Any,
    reference_b_z: Any,
    layout: ReferenceLayout,
    **kwargs: Any,
) -> tuple:
    """Materialize a sample-conditional probability table for CA fitting."""
    audit = _new_probability_audit(len(layout.sample_order))
    probabilities = np.empty(
        (int(query_e.shape[0]), int(layout.indices.size)),
        dtype=np.float32,
    )
    for start, stop, values in iter_probability_batches(
        query_e,
        reference_w_z,
        reference_b_z,
        layout,
        audit=audit,
        **kwargs,
    ):
        probabilities[start:stop] = values
    return probabilities, audit


def _fixed_dimension_selection(selected_components: int, spectrum: np.ndarray) -> dict:
    return {
        "selection_rule": "fixed_n_components",
        "selected_n_components": int(selected_components),
        "candidate_dimensions": [int(selected_components)],
        "selector_dimensions": {},
        "selector_consensus": float(selected_components),
        "n_reference_spectra": 1,
        "mean_explained_variance": np.asarray(spectrum).tolist(),
        "mean_relative_eigengap_after_component": [],
        "ground_truth_used": False,
        "cluster_count_used": False,
    }


def fit_sample_conditional_ca(
    fit_e: Any,
    all_w_z: Any,
    all_b_z: Any,
    batch_values: Sequence[Any],
    *,
    sample_order: Sequence[Any],
    e2w: torch.nn.Module,
    sample_one_hot: Mapping[str, np.ndarray],
    reference_num: int,
    reference_seed: int,
    dimension_reference_seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    candidate_dimensions: Sequence[int] = (5, 10, 20, 40),
    max_spectrum_components: int = 40,
    query_batch_size: int = 2048,
    fit_indices: Sequence[int] = (),
    device: Union[str, torch.device] = "cpu",
    n_components: Any = "auto",
) -> dict:
    """Fit sample-conditional CA from already prepared Mievformer arrays."""
    # Imported lazily to avoid a module-import cycle with the high-level API.
    from . import workflow as niche_workflow

    fit_values = np.asarray(fit_e, dtype=np.float32)
    w_z = np.asarray(all_w_z, dtype=np.float32)
    b_z = np.asarray(all_b_z, dtype=np.float32).reshape(-1)
    batches = np.asarray(batch_values).astype(str)
    reference_num = int(reference_num)
    reference_seed = int(reference_seed)
    if fit_values.ndim != 2 or fit_values.shape[0] < 3:
        raise ValueError("Sample-conditional CA requires at least three fitting cells")
    if not np.isfinite(fit_values).all():
        raise ValueError("fit_e must contain only finite values")
    if w_z.ndim != 2 or w_z.shape[0] != batches.size or b_z.shape[0] != batches.size:
        raise ValueError("w_z/b_z do not contain one row per source cell")
    if reference_num < len(tuple(sample_order)) or reference_num > batches.size:
        raise ValueError(
            "reference_num must be at least the sample count and no greater "
            "than the source-cell count"
        )
    max_rank = min(fit_values.shape[0] - 1, reference_num - 1)
    auto_dimension = isinstance(n_components, str)
    if auto_dimension and str(n_components).lower() != "auto":
        raise ValueError("n_components must be an integer or 'auto'")
    if auto_dimension:
        max_components = min(int(max_spectrum_components), max_rank)
        if max_components < 3:
            raise ValueError("Sample-conditional CA requires at least three components")
        reference_seeds = []
        for value in dimension_reference_seeds:
            value = int(value)
            if value not in reference_seeds:
                reference_seeds.append(value)
        if not reference_seeds:
            raise ValueError("dimension_reference_seeds must not be empty")
    else:
        selected_components = int(n_components)
        if not 1 <= selected_components <= max_rank:
            raise ValueError(
                f"n_components must be between 1 and {max_rank}; got {selected_components}"
            )
        max_components = selected_components
        reference_seeds = [reference_seed]

    spectra = []
    probability_audits = {}
    final_basis = None
    final_layout = None
    final_reference_w_z = None
    final_reference_b_z = None
    for seed in reference_seeds:
        layout = select_balanced_reference_indices(
            batches,
            reference_num,
            seed,
            sample_order=sample_order,
        )
        reference_w_z = w_z[layout.indices]
        reference_b_z = b_z[layout.indices]
        probabilities, probability_audit = materialize_probabilities(
            fit_values,
            reference_w_z,
            reference_b_z,
            layout,
            e2w=e2w,
            sample_one_hot=sample_one_hot,
            batch_size=int(query_batch_size),
            device=device,
        )
        basis = niche_workflow._fit_reference_probability_ca_basis(
            probabilities,
            max_components=max_components,
            device=device,
        )
        spectra.append(np.asarray(basis["explained_variance"], dtype=np.float64))
        probability_audits[f"reference_seed_{seed}"] = {
            "reference_seed": seed,
            "reference_num": int(layout.indices.size),
            "reference_unique": int(np.unique(layout.indices).size),
            "quotas": dict(layout.quotas),
            **probability_audit,
        }
        if seed == reference_seed:
            final_basis = basis
            final_layout = layout
            final_reference_w_z = reference_w_z
            final_reference_b_z = reference_b_z
        del probabilities, basis
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    spectra_array = np.vstack(spectra)
    if auto_dimension:
        selected_components, selection = (
            niche_workflow.select_reference_probability_ca_dimension(
                spectra_array,
                candidate_dimensions=tuple(int(value) for value in candidate_dimensions),
                expected_n_reference_spectra=len(reference_seeds),
            )
        )
        selection = dict(selection)
        selection["planned_reference_seeds"] = list(reference_seeds)
    else:
        selection = _fixed_dimension_selection(selected_components, spectra_array[0])

    if final_basis is None:
        final_layout = select_balanced_reference_indices(
            batches,
            reference_num,
            reference_seed,
            sample_order=sample_order,
        )
        final_reference_w_z = w_z[final_layout.indices]
        final_reference_b_z = b_z[final_layout.indices]
        probabilities, probability_audit = materialize_probabilities(
            fit_values,
            final_reference_w_z,
            final_reference_b_z,
            final_layout,
            e2w=e2w,
            sample_one_hot=sample_one_hot,
            batch_size=int(query_batch_size),
            device=device,
        )
        final_basis = niche_workflow._fit_reference_probability_ca_basis(
            probabilities,
            max_components=max(max_components, selected_components),
            device=device,
        )
        probability_audits[f"final_reference_seed_{reference_seed}"] = {
            "reference_seed": reference_seed,
            "reference_num": int(final_layout.indices.size),
            "reference_unique": int(np.unique(final_layout.indices).size),
            "quotas": dict(final_layout.quotas),
            **probability_audit,
        }

    return {
        "method": "mievformer_sample_conditional_reference_probability_ca",
        "version": 2,
        "variant": VARIANT,
        "reference_num": reference_num,
        "reference_seed": reference_seed,
        "reference_indices": final_layout.indices,
        "reference_samples": final_layout.reference_samples,
        "sample_order": np.asarray(final_layout.sample_order, dtype=str),
        "sample_one_hot": np.vstack(
            [np.asarray(sample_one_hot[sample], dtype=np.float32) for sample in final_layout.sample_order]
        ),
        "block_starts": np.asarray(
            [start for _, start, _ in final_layout.block_slices],
            dtype=np.int64,
        ),
        "block_stops": np.asarray(
            [stop for _, _, stop in final_layout.block_slices],
            dtype=np.int64,
        ),
        "reference_w_z": np.asarray(final_reference_w_z, dtype=np.float32),
        "reference_b_z": np.asarray(final_reference_b_z, dtype=np.float32),
        "column_mass": np.asarray(final_basis["column_mass"], dtype=np.float32),
        "components": np.asarray(
            final_basis["components"][:, :selected_components], dtype=np.float32
        ),
        "components_full": np.asarray(final_basis["components"], dtype=np.float32),
        "selected_n_components": int(selected_components),
        "explained_variance": np.asarray(
            final_basis["explained_variance"], dtype=np.float64
        ),
        "explained_variance_ratio": np.asarray(
            final_basis["explained_variance_ratio"], dtype=np.float64
        ),
        "total_variance": float(final_basis["total_variance"]),
        "negative_eigenvalue_count": int(final_basis["negative_eigenvalue_count"]),
        "fit_indices": np.asarray(fit_indices, dtype=np.int64),
        "dimension_reference_seeds": np.asarray(reference_seeds, dtype=np.int64),
        "reference_spectra": spectra_array,
        "dimension_selection": selection,
        "fit_probability_audits": probability_audits,
        "query_batch_size": int(query_batch_size),
        "fit_device": str(torch.device(device)),
    }


def layout_from_model(model: Mapping[str, Any]) -> ReferenceLayout:
    """Reconstruct and validate a balanced reference layout from an artifact."""
    order = tuple(str(value) for value in np.asarray(model["sample_order"]))
    starts = np.asarray(model["block_starts"], dtype=np.int64)
    stops = np.asarray(model["block_stops"], dtype=np.int64)
    indices = np.asarray(model["reference_indices"], dtype=np.int64)
    reference_samples = np.asarray(model["reference_samples"]).astype(str)
    if starts.size != len(order) or stops.size != len(order):
        raise ValueError("Saved CA model has invalid block metadata")
    blocks = tuple(
        (sample, int(start), int(stop))
        for sample, start, stop in zip(order, starts, stops)
    )
    if not blocks or blocks[0][1] != 0 or blocks[-1][2] != indices.size:
        raise ValueError("Saved CA model blocks do not cover its references")
    if any(start >= stop for _, start, stop in blocks):
        raise ValueError("Saved CA model contains an empty reference block")
    if any(left[2] != right[1] for left, right in zip(blocks[:-1], blocks[1:])):
        raise ValueError("Saved CA model reference blocks are not contiguous")
    if reference_samples.size != indices.size:
        raise ValueError("Saved CA model reference labels have invalid length")
    return ReferenceLayout(
        indices=indices,
        reference_samples=reference_samples,
        sample_order=order,
        quotas={sample: stop - start for sample, start, stop in blocks},
        block_slices=blocks,
        seed=int(model["reference_seed"]),
    )


def project_probability_rows(
    probabilities: Any,
    column_mass: Any,
    components: Any,
) -> np.ndarray:
    """Project probability rows into an already fitted CA basis."""
    values = np.asarray(probabilities, dtype=np.float32).copy()
    mass = np.asarray(column_mass, dtype=np.float32).reshape(-1)
    basis = np.asarray(components, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != mass.size:
        raise ValueError("Probability table and CA column mass are incompatible")
    if basis.ndim != 2 or basis.shape[0] != mass.size:
        raise ValueError("CA components and column mass are incompatible")
    if np.any(mass <= 0) or not np.isfinite(mass).all():
        raise ValueError("CA column masses must be finite and positive")
    values -= mass[None, :]
    values *= np.reciprocal(np.sqrt(mass)).astype(np.float32)[None, :]
    scores = values @ basis
    if not np.isfinite(scores).all():
        raise ValueError("Sample-conditional CA scores are non-finite")
    return np.asarray(scores, dtype=np.float32)


def transform_sample_conditional_ca(
    query_e: Any,
    model: Mapping[str, Any],
    *,
    e2w: torch.nn.Module,
    sample_one_hot: Mapping[str, np.ndarray] = None,
    batch_size: int = None,
    device: Union[str, torch.device] = "cpu",
) -> tuple:
    """Project query embeddings with a frozen sample-conditional CA artifact."""
    layout = layout_from_model(model)
    if sample_one_hot is None:
        matrix = np.asarray(model.get("sample_one_hot"), dtype=np.float32)
        if matrix.shape != (len(layout.sample_order), len(layout.sample_order)):
            raise ValueError("Saved CA model lacks a valid sample_one_hot mapping")
        sample_one_hot = {
            sample: matrix[index]
            for index, sample in enumerate(layout.sample_order)
        }
    if batch_size is None:
        batch_size = int(model.get("query_batch_size", 2048))
    n_components = int(model["selected_n_components"])
    components = np.asarray(
        model.get("components", model.get("components_full")), dtype=np.float32
    )[:, :n_components]
    scores = np.empty(
        (int(query_e.shape[0]), n_components),
        dtype=np.float32,
    )
    audit = _new_probability_audit(len(layout.sample_order))
    for start, stop, probabilities in iter_probability_batches(
        query_e,
        model["reference_w_z"],
        model["reference_b_z"],
        layout,
        e2w=e2w,
        sample_one_hot=sample_one_hot,
        batch_size=int(batch_size),
        device=device,
        audit=audit,
    ):
        scores[start:stop] = project_probability_rows(
            probabilities,
            model["column_mass"],
            components,
        )
    return scores, audit


def model_array_payload(model: Mapping[str, Any]) -> dict:
    """Return the numerical members needed for an NPZ projection artifact."""
    keys: Iterable[str] = (
        "reference_indices",
        "reference_obs_names",
        "reference_samples",
        "sample_order",
        "sample_one_hot",
        "block_starts",
        "block_stops",
        "reference_w_z",
        "reference_b_z",
        "column_mass",
        "components",
        "components_full",
        "explained_variance",
        "explained_variance_ratio",
        "fit_indices",
        "dimension_reference_seeds",
        "reference_spectra",
    )
    payload = {key: np.asarray(model[key]) for key in keys if key in model}
    payload["selected_n_components"] = np.asarray(
        [int(model["selected_n_components"])],
        dtype=np.int64,
    )
    return payload
