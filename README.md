# mievformer

mievformer is a Python package for learning cellular microenvironments from
spatial transcriptomics. Its standard workflow uses reference-probability
correspondence analysis (CA) for niche clustering and UMAP, with
sample-conditional CA for joint analysis of multiple spatial slices.

## Installation

```bash
pip install mievformer
```

## Documentation

For detailed usage instructions, tutorials, and API reference, please visit the documentation:

**https://kojikoji.github.io/mievformer_package/index.html**

## Quick start

~~~python
import mievformer as mf

# Single slice: ordinary reference-probability CA.
adata = mf.optimize_nicheformer(adata, model_path="model.pth")

# Multiple slices: batch-conditioned training and sample-conditional CA.
adata = mf.optimize_nicheformer(
    adata,
    model_path="multibatch_model.pth",
    batch_key="sample",
)
~~~

## License

MIT License
