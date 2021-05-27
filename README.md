# metaflow-helper

Convenience utilities for common machine learning tasks on Metaflow

[![PyPI version](https://badge.fury.io/py/metaflow-helper.svg)](https://badge.fury.io/py/metaflow-helper)

| Unit tests | Examples |
| ---------- | -------  | 
| ![Build](https://github.com/fwhigh/metaflow-helper/actions/workflows/python36.yml/badge.svg) | ![Build](https://github.com/fwhigh/metaflow-helper/actions/workflows/examples36.yml/badge.svg) |
| ![Build](https://github.com/fwhigh/metaflow-helper/actions/workflows/python37.yml/badge.svg) | ![Build](https://github.com/fwhigh/metaflow-helper/actions/workflows/examples37.yml/badge.svg) |
| ![Build](https://github.com/fwhigh/metaflow-helper/actions/workflows/python38.yml/badge.svg) | ![Build](https://github.com/fwhigh/metaflow-helper/actions/workflows/examples38.yml/badge.svg) |

## Quickstart

You can run the model selection tournament immediately like this. 
Install a convenience package called metaflow-helper.


```bash
git clone https://github.com/fwhigh/metaflow-helper.git
cd metaflow-helper
python -m pip install .
```

Then run a Metaflow example. 
Examples need a few more packages, including Metaflow itself, which metaflow-helper doesn’t currently require.

```bash
python -m pip install -r example-requirements.txt
python examples/model-selection/train.py run
```

You can visualize the example flow.

```bash
python examples/model-selection/train.py output-dot | dot -Grankdir=TB -Tpng -o model-selection-flow.png
```

## Release procedure

1. Bump VERSION in setup.py. This should contain the new version without the v
1. Run `git tag -a v0.0.1 -m "Release v0.0.1"` with VERSION version number
1. Run `git commit -a -m "Release v0.0.1"` with VERSION version number
1. Optional: run `make package`
1. Run `git push origin v0.0.1`
