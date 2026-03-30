# Installation

## Requirements

Mievformer requires Python 3.9 or later. The following dependencies will be installed automatically:

- PyTorch >= 2.0
- NumPy (<2.0 recommended for PyTorch compatibility)
- Pandas
- Scanpy
- AnnData
- PyTorch Lightning
- scikit-learn
- matplotlib
- seaborn
- scipy

### Optional Dependencies

For full functionality, the following packages are recommended:

- **igraph** and **leidenalg**: Required for Leiden clustering in Scanpy
- **tensorboard**: Required for training visualization and logging

```bash
pip install igraph leidenalg tensorboard
```

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

## Docker Installation

For a reproducible environment with all dependencies pre-configured, you can use Docker:

### Building the Docker Image

```bash
cd mievformer_package
docker build -t mievformer .
```

### Running with GPU Support

Requires NVIDIA Container Toolkit to be installed on the host system.

```bash
docker run --gpus all -it mievformer
```

### Running the Tutorial

```bash
docker run --gpus all mievformer python run_tutorial.py
```

### Interactive Mode

```bash
docker run --gpus all -it mievformer bash
```

## Troubleshooting

### NumPy Compatibility

If you encounter errors related to NumPy 2.x incompatibility with PyTorch, downgrade NumPy:

```bash
pip install "numpy<2"
```

### Missing igraph/leidenalg

If Leiden clustering fails with an import error, install the required packages:

```bash
pip install igraph leidenalg
```

### TensorBoard Not Found

If training fails with a TensorBoard error, install it:

```bash
pip install tensorboard
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
