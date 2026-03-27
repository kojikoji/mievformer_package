# Installation

## Requirements

Mievformer requires Python 3.9 or later. The following dependencies will be installed automatically:

- PyTorch >= 2.0
- NumPy
- Pandas
- Scanpy
- AnnData
- PyTorch Lightning
- scikit-learn
- matplotlib
- seaborn
- scipy

## Installation via pip

The recommended way to install Mievformer is via pip:

```bash
pip install mievformer
```

## Installation from source

To install the latest development version from source:

```bash
git clone https://github.com/kojikoji/mievformer_package.git
cd mievformer_package
pip install -e .
```

## GPU Support

Mievformer can leverage GPU acceleration for training. To enable GPU support, ensure you have CUDA installed and install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install Mievformer:

```bash
pip install mievformer
```

## Verifying Installation

To verify that Mievformer is installed correctly:

```python
import mievformer as mf
print(mf.__all__)
```

This should print the list of available functions:

```
['optimize_nicheformer', 'calculate_wb_ez', 'calculate_spatial_distribution',
 'aggregate_dist_e', 'estimate_population_density', 'analyze_density_correlation',
 'analyze_niche_composition']
```
