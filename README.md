# metalearn

Convenience utilities for common machine learning tasks on Metaflow

![Build](https://github.com/fwhigh/metalearn/actions/workflows/python36.yml/badge.svg)
![Build](https://github.com/fwhigh/metalearn/actions/workflows/python37.yml/badge.svg)
![Build](https://github.com/fwhigh/metalearn/actions/workflows/python38.yml/badge.svg)

## Quickstart

Install the package.

```bash
git clone https://github.com/fwhigh/metalearn.git
cd metalearn
python -m pip install .
```

Run the examples.

```bash
python -m pip install example-requirements.txt
python examples/model-tournament/train.py
```

## Release procedure

1. Bump VERSION in setup.py. This should contain the new version without the v
1. Run `git commit -a -m "Release v0.0.1"`
1. Run `git tag -a v0.0.1 -m "Release v0.0.1"` with matching version number
1. Optional: run `make package`
1. Run `git push origin v0.0.1`
