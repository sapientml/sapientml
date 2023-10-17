![SapientML](https://raw.githubusercontent.com/sapientml/sapientml/main/static/SapientML_positive_logo.svg#gh-light-mode-only)
![](./static/SapientML_negative_logo.svg#gh-dark-mode-only)
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
<a href="https://codecov.io/gh/sapientml/sapientml" ><img src="https://codecov.io/gh/sapientml/sapientml/graph/badge.svg?token=STVPNF5X25"/></a>
<a href="https://pepy.tech/project/sapientml"><img src="https://static.pepy.tech/badge/sapientml"/></a>
<a href="https://pepy.tech/project/sapientml"><img src="https://static.pepy.tech/badge/sapientml/month"/></a>
</p>

# Installation

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
# Getting Started

Please see our [Documentation](https://sapientml.readthedocs.io/en/latest/user/usage.html) for further details.
## Run AutoML

```py
import pandas as pd
from sapientml import SapientML
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("https://github.com/sapientml/sapientml/files/12481088/titanic.csv")
train_data, test_data = train_test_split(train_data)
y_true = test_data["survived"].reset_index(drop=True)
test_data.drop(["survived"], axis=1, inplace=True)

cls = SapientML(["survived"])

cls.fit(train_data)
y_pred = cls.predict(test_data)

y_pred = y_pred["survived"].rename("survived_pred")
print(f"F1 score: {f1_score(y_true, y_pred)}")
```

## Obtain and Run Generated Code

You can access `model` field to get a model consisting of generated code after executing `fit` method.
`model` provides `fit`, `predict`, and `save` method to train a model by generated code, predict from a test data by generated code, and save generated code to a designated folder.

```py
model = sml.fit(train_data, codegen_only=True).model

model.fit(X_train, y_train) # build a model by using another data and the same generated code

y_pred = model.predict(X_test) # prediction by using generated code

model.save("/path/to/output") # save generated code to `path/to/output`
```

# Examples

| Dataset                                                                                                            | Task             | Target      | Code                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------ | ---------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Titanic Dataset](https://www.openml.org/d/40945)                                                                  | `classification` | `survived`  | <a target="_blank" href="https://colab.research.google.com/github/sapientml/sapientml/blob/main/static/sapientml-example-titanic.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                      |
| Hotel Cancellation                                                                                                 | `classification` | `Status`    | <a target="_blank" href="https://colab.research.google.com/github/sapientml/sapientml/blob/main/static/sapientml-example-hotel-candel-prediction.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>      |
| Housing Prices                                                                                                     | `regression`     | `SalePrice` | <a target="_blank" href="https://colab.research.google.com/github/sapientml/sapientml/blob/main/static/sapientml-example-housing-prices.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>               |
| [Medical Insurance Charges](https://www.kaggle.com/datasets/harishkumardatalab/medical-insurance-price-prediction) | `regression`     | `charges`   | <a target="_blank" href="https://colab.research.google.com/github/sapientml/sapientml/blob/main/static/sapientml-example-medical-insurance-prediction.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

# Publications

The technologies of the software originates from the following research paper published at the International Conference on Software Engineering (ICSE), which is one of the premier conferences on Software Engineering.

**Ripon K. Saha, Akira Ura, Sonal Mahajan, Chenguang Zhu, Linyi Li, Yang Hu, Hiroaki Yoshida, Sarfraz Khurshid, Mukul R. Prasad (2022, May). [SapientML: Synthesizing Machine Learning Pipelines by Learning from Human-Written Solutions](https://arxiv.org/abs/2202.10451). In *[Proceedings of the 44th International Conference on Software Engineering](https://conf.researchr.org/home/icse-2022)* (pp. 1932-1944).**

```bibtex
@inproceedings{10.1145/3510003.3510226,
author = {Saha, Ripon K. and Ura, Akira and Mahajan, Sonal and Zhu, Chenguang and Li, Linyi and Hu, Yang and Yoshida, Hiroaki and Khurshid, Sarfraz and Prasad, Mukul R.},
title = {SapientML: Synthesizing Machine Learning Pipelines by Learning from Human-Written Solutions},
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
