![sapientml_logo](https://github.com/sapientml/sapientml/assets/1414384/05dfb90d-d5a1-40fa-b1fd-ded938680aaa)
### Generative AutoML for Tabular Data

[![PyPI version](https://badge.fury.io/py/sapientml.svg)](https://badge.fury.io/py/sapientml) ![Static Badge](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue) [![Release](https://github.com/sapientml/sapientml/actions/workflows/release.yml/badge.svg)](https://github.com/sapientml/sapientml/actions/workflows/release.yml) [![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org) [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/7781/badge)](https://www.bestpractices.dev/projects/7781)

SapientML is an 

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


