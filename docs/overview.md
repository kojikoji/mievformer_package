# Overview of Mievformer

## Abstract

Spatial omics has now reached single-cell resolution, enabling the joint analysis of tissue microenvironments and cellular-state heterogeneity. This capability opens the door to systematic comparisons of how cells differ across local contexts within a tissue. However, key challenges remain: how to computationally define a microenvironmental state at single-cell resolution, and which representations are most informative for biological discovery.

Here, we present **Mievformer**, a Transformer-based masked self-supervised framework that learns microenvironmental embeddings parameterizing the conditional distribution of cellular states at a masked central position. Across diverse simulation settings, Mievformer outperforms existing methods in accuracy. On multiple real datasets, using an unsupervised (ground-truth-free) evaluation metric that showed the strongest correlation with ground-truth metrics in simulations, Mievformer consistently exceeds competing approaches.

By explicitly modelling the joint structure of microenvironmental and cellular states, Mievformer enables analyses beyond conventional clustering, including:
1. Discovery of cellular subpopulations based on their distribution across microenvironments.
2. Identification of gene-expression signatures associated with colocalization with specific cell populations.

Together, these results establish Mievformer as a quantitatively superior and biologically informative framework for learning microenvironment representations and dissecting the coupling between cellular heterogeneity and tissue microenvironments.

## Methodology

### Input Data Preparation and Model Optimization

Mievformer takes as input spatial transcriptome data comprising gene expression vectors $y_i$ and spatial coordinates $x_i$ for each cell $i$, and computes microenvironmental embeddings $e_i$ representing the local microenvironmental state. For computational efficiency and optimization stability, rather than operating directly on gene expression, Mievformer uses low-dimensional representations $z_i$ obtained via principal component analysis (PCA).

The core optimization process is performed by {func}`mievformer.optimize_nicheformer`. This function handles:
*   Preprocessing of spatial coordinates.
*   Construction of the dataset with masked central cells.
*   Training of the Transformer-based model using masked self-supervised learning.
*   Computation of microenvironmental embeddings $e_i$.

For each cell $i$, the embedding $e_i$ is computed from the cell states $\{z_j \mid j \in N(i)\}$ within its spatial neighborhood $N(i)$. The neighborhood is determined using k-nearest neighbors (default 100). Spatial coordinates are encoded as relative positions and transformed into positional encodings via sinusoidal encoding.

### Algorithm

Mievformer employs a masked self-supervised learning framework to compute the probability of observing a central cell state given its surrounding cellular context, and maximizes this probability during training. The architecture comprises two main components: **NicheEncoder** and **Distributor**.

*   **NicheEncoder**: Constructs the input sequence by placing a learnable mask token at the central cell position. It processes this sequence through multiple Transformer encoder layers. The final hidden state of the mask token is mapped to the mean $\mu_i$ and standard deviation $\sigma_i$ of a Gaussian distribution over microenvironmental embeddings.
*   **Distributor**: Computes the probability that each cell embedding $z_j$ in the mini-batch could be observed at the central position given microenvironment $e_i$.

The training objective maximizes the likelihood that the observed central cell state $z_i$ is generated from its inferred microenvironment $e_i$. This formulation corresponds to the InfoNCE loss, maximizing mutual information between paired observations.

After training, {func}`mievformer.calculate_wb_ez` can be used to calculate the embeddings required for the score function ($w_e$, $w_z$, $b_z$) and add them to the AnnData object.

### Downstream Analyses

Mievformer enables several downstream analyses:

1.  **Microenvironmental Clustering**: Characterization of microenvironmental heterogeneity. This can be performed and visualized using {func}`mievformer.analyze_niche_composition`.
2.  **Cell Subpopulation Identification**: By applying Bayes' theorem to the learned conditional probability $P(z|e)$, we obtain $P(e|z)$, the posterior distribution over microenvironments given a cell state. Aggregating these probabilities yields single-cell-resolution microenvironmental membership profiles. This is facilitated by {func}`mievformer.calculate_spatial_distribution` and {func}`mievformer.aggregate_dist_e`.
3.  **Colocalization Analysis**: By integrating $P(z|e)$ over all cell states belonging to a specific cell population, we obtain the density (existence probability) of that population in microenvironment $e$. This is computed by {func}`mievformer.estimate_population_density`. Correlating this density estimate with gene expression reveals expression signatures associated with colocalization, which can be analyzed using {func}`mievformer.analyze_density_correlation`.
