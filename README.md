# RAD Research Public Repo

This repository contains the code for the papers published with the Risk Aware Driving (RAD) team at Toyota Research Institute (TRI).


## Overview

This repo contains the following packages and folders:

* `data_sources` - Converters for different data sources. 
* `intent` - Training and experiment scripts.
* `model_zoo` - The pytorch models.
* `radutils` - RAD research utilities.
* `triceps` - TRI Common Environment and Prediction Serializer.

See [prediction_framework_overview.pdf](prediction_framework_overview.pdf) for additional details.

See below in order to run the code for each paper under `intent/multiagents/`:
* [`hybrid`](./intent/multiagents/hybrid/README.md) - Code for "HYPER: Learned Hybrid Trajectory Prediction via Factored Inference and Adaptive Sampling"
* [`language`](./intent/multiagents/language/README.md) - Code for "Trajectory Prediction with Linguistic Representations"

# Development environment

Conda is used to create the development environment.

* To create the environment using existing conda.
```bash
conda env create -f environment.pt190.yml
```

__OR__

* Let it download miniconda to ~/miniconda3
```bash
./scripts/create_env.sh --skip-aws
```

__Then__

Activate the conda environment, before running the code.

```bash
conda activate pt190
```

## License
See the `LICENSE.md` file for details.

Copyright 2018-2022 TRI.
