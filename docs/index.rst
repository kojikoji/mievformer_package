Mievformer - Microenvironment Inference via Transformer
========================================================

**Mievformer** is a Transformer-based masked self-supervised framework for learning microenvironmental embeddings from spatial transcriptomics data. It enables the discovery of cellular subpopulations based on their distribution across microenvironments and identification of gene-expression signatures associated with cell colocalization.

Key Features
------------

- **Microenvironmental Embedding**: Learn representations that parameterize the conditional distribution of cellular states at a masked central position.
- **Superior Accuracy**: Outperforms existing methods across diverse simulation settings and real datasets.
- **Downstream Analyses**: Enables microenvironmental clustering, cell subpopulation identification, and colocalization analysis.
- **Seamless Integration**: Works with AnnData objects and integrates with the scanpy ecosystem.

Getting Started
---------------

Install Mievformer via pip:

.. code-block:: bash

   pip install mievformer

Basic usage example:

.. code-block:: python

   import mievformer as mf
   import scanpy as sc

   # Load your spatial transcriptomics data
   adata = sc.read_h5ad("your_data.h5ad")

   # Train the model and compute microenvironmental embeddings
   adata = mf.optimize_nicheformer(adata, model_path="model.pth")

   # Calculate embeddings for downstream analysis
   adata = mf.calculate_wb_ez(adata, "model.pth")

   # Compute niche density ratio and per-cell niche-cluster membership
   adata = mf.calculate_niche_density_ratio(adata)
   adata = mf.calculate_niche_cluster_membership(adata)

   # Cluster cells by niche membership and visualize
   adata = mf.analyze_niche_membership(adata, file_path="niche_composition.png")


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   tutorials/getting_started

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   overview
   api

.. toctree::
   :maxdepth: 1
   :caption: About

   release_notes
   references


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
