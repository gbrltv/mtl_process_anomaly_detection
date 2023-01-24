# Meta-learning for Anomaly Detection in Process Mining

> This file lists the steps to reproduce the experiments, analysis and figures generated for the paper [Process Mining Encoding via Meta-learning for an Enhanced Anomaly Detection](https://link.springer.com/chapter/10.1007/978-3-030-85082-1_15).


## Contents

This repository already comes with the datasets employed in the experiments along with the code to reproduce them. We also provide the experimental results (see *.csv* files) and figures used in the papers (see *plots* folder).


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

```shell
conda install -c conda-forge openjdk
```


## Reproducing experiments

The first step is to extract the meta-features from the event logs:

```shell
python compute_encoding/extract_meta_features.py
```

Then, we compute the encodings of each method:

```shell
python compute_encoding/doc2vec.py
python compute_encoding/node2vec_.py
python compute_encoding/alignment.py
```

The encodings are then submitted to a classification pipeline and metrics regarding their performance are recorded:

```shell
python compute_encoding/classification.py
```

The last step creates the analysis and plots the results in a dedicated folder:

```shell
python compute_encoding/analysis.py
```
