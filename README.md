![](./static/logo_SapientML_negative@2x.png#gh-dark-mode-only)![](./static/logo_SapientML_positive@2x.png#gh-light-mode-only)
<div style="text-align:center;">
<h1>Generative AutoML for Tabular Data</h1>
<a href="https://badge.fury.io/py/sapientml"><img src="https://badge.fury.io/py/sapientml.svg" alt="PyPI version"></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/sapientml">
<a href="https://github.com/sapientml/sapientml/actions/workflows/release.yml"><img alt="Release" src="https://github.com/sapientml/sapientml/actions/workflows/release.yml/badge.svg"></a>
<a href="https://conventionalcommits.org"><img alt="Conventional Commits" src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white"></a>
<a href="https://www.bestpractices.dev/projects/7781"><img alt="OpenSSF Best Practices" src="https://www.bestpractices.dev/projects/7781/badge"></a><br />
SapientML is an AutoML technology that can learn from a corpus of existing datasets and their human-written pipelines, and efficiently generate a high-quality pipeline for a predictive task on a new dataset.
</div>

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

### Generate Code in a notebook

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

### Run Generated Code

```
cd outputs/
python final_script.py
```


