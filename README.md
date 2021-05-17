# metaflow-helper

Convenience utilities for common machine learning tasks on Metaflow

![Build](https://github.com/fwhigh/metaflow-helper/actions/workflows/python36.yml/badge.svg)
![Build](https://github.com/fwhigh/metaflow-helper/actions/workflows/python37.yml/badge.svg)
![Build](https://github.com/fwhigh/metaflow-helper/actions/workflows/python38.yml/badge.svg)

![Build](https://github.com/fwhigh/metaflow-helper/actions/workflows/examples36.yml/badge.svg)
![Build](https://github.com/fwhigh/metaflow-helper/actions/workflows/examples37.yml/badge.svg)
![Build](https://github.com/fwhigh/metaflow-helper/actions/workflows/examples38.yml/badge.svg)

## Quickstart

You can run the tournament immediately like this. 
First, install a convenience package I'm calling `metaflow-helper`.

```bash
git clone https://github.com/fwhigh/metaflow-helper.git
cd metaflow-helper
python -m pip install .
```

Second, run the Metaflow tournament job. 
This one needs a few more packages, including Metaflow itself, 
which `metaflow-helper` doesn't currently require.

```bash
python -m pip install -r example-requirements.txt
python examples/model-selection/train.py run
```

Visualize the flow.

```bash
python examples/model-selection/train.py output-dot | dot -Grankdir=TB -Tpng -o flow.pngflow.png
```

## Release procedure

1. Bump VERSION in setup.py. This should contain the new version without the v
1. Run `git commit -a -m "Release v0.0.1"`
1. Run `git tag -a v0.0.1 -m "Release v0.0.1"` with matching version number
1. Optional: run `make package`
1. Run `git push origin v0.0.1`
