<h1 align="center">
<img alt="SapientML" src="./static/logo_SapientML_positive%402x.png#gh-light-mode-only">
<img alt="" src="./static/logo_SapientML_negative%402x.png#gh-dark-mode-only">
</h1>
<h1 align="center">Generative AutoML for Tabular Data</h1>
<p align='center'>
SapientML is an AutoML technology that can learn from a corpus of existing datasets and their human-written pipelines, and efficiently generate a high-quality pipeline for a predictive task on a new dataset.
</p>
<p align='center'>
<a href="https://badge.fury.io/py/sapientml"><img src="https://badge.fury.io/py/sapientml.svg" alt="PyPI version"></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/sapientml">
<a href="https://github.com/sapientml/sapientml/actions/workflows/release.yml"><img alt="Release" src="https://github.com/sapientml/sapientml/actions/workflows/release.yml/badge.svg"></a>
<a href="https://conventionalcommits.org"><img alt="Conventional Commits" src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white"></a>
<a href="https://www.bestpractices.dev/projects/7781"><img alt="OpenSSF Best Practices" src="https://www.bestpractices.dev/projects/7781/badge"></a>
</p>

# Getting Started

## Installation

From PyPI repository

```
pip install sapientml
```

From source code:

```
git clone https://github.com/sapientml/sapientml.git
cd sapientml
pip install poetry
poetry install
```

### Run AutoML

<a target="_blank" href="https://colab.research.google.com/github/sapientml/sapientml/blob/feature/documentation/static/sapientml-example-titanic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```py
import pandas as pd
from sapientml import SapientML
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("https://github.com/sapientml/sapientml/files/12481088/titanic.csv")
train_data, test_data = train_test_split(train_data)
y_true = test_data["survived"].reset_index(drop=True)
test_data.drop(["survived"], axis=1, inplace=True)

sml = SapientML(["survived"])

sml.fit(train_data)
y_pred = sml.predict(test_data)

print(f"F1 score: {f1_score(y_true, y_pred)}")
```

## Running Generated Code Manually

You can get generated code in the output folder after executing `fit` method.

### Hold-out Validation

Run `outputs/final_script.py`, then you will see a result of the hold-out validation using the train data

```
cd outputs/
python final_script.py
```

### Train a Model by Generated Code

Run `outputs/final_train.py`, then you will get several `.pkl` files containing a trained model and some components for preprocessing.

```
cd outputs/
python final_train.py
```

### Prediction by using Trained Model

Run `outputs/final_predict.py` with `outputs/test.pkl` exist already or prepared manually if not exist. `test.pkl` must contain a `pandas.DataFrame` object created from a CSV file fto be predited.

```
cd outputs/
python final_predict.py
```

# Publications

The technologies of the software originates from the following research paper published at the International Conference on Software Engineering (ICSE), which is one of the premier conferences on Software Engineering.

**Ripon K. Saha, Akira Ura, Sonal Mahajan, Chenguang Zhu, Linyi Li, Yang Hu, Hiroaki Yoshida, Sarfraz Khurshid, Mukul R. Prasad (2022, May). [SapientML: synthesizing machine learning pipelines by learning from human-writen solutions](https://arxiv.org/abs/2202.10451). In *[Proceedings of the 44th International Conference on Software Engineering](https://conf.researchr.org/home/icse-2022)* (pp. 1932-1944).**

```bibtex
@inproceedings{10.1145/3510003.3510226,
author = {Saha, Ripon K. and Ura, Akira and Mahajan, Sonal and Zhu, Chenguang and Li, Linyi and Hu, Yang and Yoshida, Hiroaki and Khurshid, Sarfraz and Prasad, Mukul R.},
title = {SapientML: Synthesizing Machine Learning Pipelines by Learning from Human-Writen Solutions},
year = {2022},
isbn = {9781450392211},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3510003.3510226},
doi = {10.1145/3510003.3510226},
abstract = {Automatic machine learning, or AutoML, holds the promise of truly democratizing the use of machine learning (ML), by substantially automating the work of data scientists. However, the huge combinatorial search space of candidate pipelines means that current AutoML techniques, generate sub-optimal pipelines, or none at all, especially on large, complex datasets. In this work we propose an AutoML technique SapientML, that can learn from a corpus of existing datasets and their human-written pipelines, and efficiently generate a high-quality pipeline for a predictive task on a new dataset. To combat the search space explosion of AutoML, SapientML employs a novel divide-and-conquer strategy realized as a three-stage program synthesis approach, that reasons on successively smaller search spaces. The first stage uses meta-learning to predict a set of plausible ML components to constitute a pipeline. In the second stage, this is then refined into a small pool of viable concrete pipelines using a pipeline dataflow model derived from the corpus. Dynamically evaluating these few pipelines, in the third stage, provides the best solution. We instantiate SapientML as part of a fully automated tool-chain that creates a cleaned, labeled learning corpus by mining Kaggle, learns from it, and uses the learned models to then synthesize pipelines for new predictive tasks. We have created a training corpus of 1,094 pipelines spanning 170 datasets, and evaluated SapientML on a set of 41 benchmark datasets, including 10 new, large, real-world datasets from Kaggle, and against 3 state-of-the-art AutoML tools and 4 baselines. Our evaluation shows that SapientML produces the best or comparable accuracy on 27 of the benchmarks while the second best tool fails to even produce a pipeline on 9 of the instances. This difference is amplified on the 10 most challenging benchmarks, where SapientML wins on 9 instances with the other tools failing to produce pipelines on 4 or more benchmarks.},
booktitle = {Proceedings of the 44th International Conference on Software Engineering},
pages = {1932–1944},
numpages = {13},
keywords = {AutoML, program synthesis, program analysis, machine learning},
location = {Pittsburgh, Pennsylvania},
series = {ICSE '22}
}
```
