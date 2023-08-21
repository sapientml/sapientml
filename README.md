# SapientML


[![PyPI version](https://badge.fury.io/py/sapientml.svg)](https://badge.fury.io/py/sapientml) ![Static Badge](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue) [![Release](https://github.com/sapientml/sapientml/actions/workflows/release.yml/badge.svg)](https://github.com/sapientml/sapientml/actions/workflows/release.yml) [![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)


# Getting Started

## Installation

```
pip install sapientml
```

### Generate Code in a notebook

Please download [housing-prices.csv](https://github.com/sapientml/sapientml/files/12374429/housing-prices.csv) to execute the following code.

```py
import pandas as pd
from sapientml import SapientML
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("housing-prices.csv")
train_data, test_data = train_test_split(train_data)
test_data.drop(["SalePrice"], axis=1)

sml = SapientML(
    ["SalePrice"],
    adaptation_metric="RMSE",
    id_columns_for_prediction=["Id"],
)

sml.fit(train_data, ignore_columns=["Id"])
pred = sml.predict(test_data)

print(pred)
```

### Run Generated Code

```
cd outputs/
python final_script.py
```


