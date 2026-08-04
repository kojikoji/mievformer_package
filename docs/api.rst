API Reference
=============

This page provides a comprehensive reference for all functions in the Mievformer package, organized by functionality.

Core Model Functions
--------------------

Functions for training the Mievformer model and computing embeddings.

``optimize_nicheformer`` uses reference-probability CA by default. Single-slice
data use ordinary CA; multiple values in ``batch_key`` use sample-conditional
CA and automatically enable batch-conditioned training. The raw neural-network
embedding remains available in ``obsm['mievformer_raw_e']``.

.. autofunction:: mievformer.optimize_nicheformer

.. autofunction:: mievformer.calculate_wb_ez


Niche Density Ratio and Membership
----------------------------------

Functions for computing the per-cell niche density ratio p(e|z)/p(e) and
aggregating it into a per-cell soft membership over niche clusters.

.. autofunction:: mievformer.calculate_niche_density_ratio

.. autofunction:: mievformer.calculate_niche_cluster_membership


Downstream Analysis
-------------------

Functions for biological interpretation and visualization.

.. autofunction:: mievformer.estimate_population_density

.. autofunction:: mievformer.analyze_density_correlation

.. autofunction:: mievformer.analyze_niche_membership
