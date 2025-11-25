# mievformer

mievformer is a Python package for nicheformer analysis, providing tools for spatial transcriptomics data analysis, including niche modeling, cell-cell interaction estimation, and visualization.

## Installation

```bash
pip install .
```

## Usage

See the tutorial notebook for detailed usage examples.

```python
import mievformer as mf
import scanpy as sc

# Load your data
adata = sc.read_h5ad("path/to/your/data.h5ad")

# Optimize model
adata = mf.optimize_nicheformer(adata, ...)
```
