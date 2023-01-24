# AutoML code for reproducibility

> This file lists the steps to reproduce the experiments, analysis and figures generated for the paper "AAA" found HERE.


## Contents

<!-- This repository already comes with the datasets employed in the experiments along with the code to reproduce them. We also provide the experimental results (see *.csv* files) and figures used in the papers (see *analysis* folder). -->


## Installation steps

First, you need to install conda to manage the environment. See installation instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

The next step is to create the environment. For that, run:

```shell
conda create --name mtl_anomaly python=3.7.0
```

Then, activate the environment:

```shell
conda activate mtl_anomaly
```

Finally, install the dependencies:

```shell
python -m pip install -r requirements.txt
```


## Reproducing experiments

```shell
python compute_encoding/extract_meta_features.py

python compute_encoding/doc2vec.py
python compute_encoding/node2vec_.py
python compute_encoding/alignment.py

python compute_encoding/classification.py

python compute_encoding/analysis.py
```
