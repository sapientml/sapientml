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
train_data, test_data = train_test_split(train_data, train_size=0.9)
y_test = test_data["SalePrice"].reset_index(drop=True)
test_data.drop(["SalePrice"], axis=1)

sml = SapientML(["SalePrice"])

sml.fit(train_data)
y_pred = sml.predict(test_data)

print(pd.concat([y_pred["SalePrice"].rename("SalePrice_pred"), y_test], axis=1))
```

```
[2023-08-22 17:02:26] INFO:Loading dataset...
[2023-08-22 17:02:26] WARNING:Metric is not specified. Use 'r2' by default.
[2023-08-22 17:02:26] INFO:Generating pipelines...
[2023-08-22 17:02:26] INFO:Generating meta features ...
[2023-08-22 17:02:28] INFO:Executing generated pipelines...
[2023-08-22 17:02:28] INFO:Running script (1/3) ...
[2023-08-22 17:02:32] INFO:Running script (2/3) ...
[2023-08-22 17:02:33] INFO:Running script (3/3) ...
[2023-08-22 17:02:35] INFO:Evaluating execution results of generated pipelines...
[2023-08-22 17:02:35] INFO:Done.
[2023-08-22 17:02:35] INFO:Building model by generated pipeline...
[2023-08-22 17:02:38] INFO:Done.
[2023-08-22 17:02:38] INFO:Predicting by built model...
     SalePrice_pred  SalePrice
0     121532.531098     104900
1      96658.207729      82500
2     217862.132461     193000
3     249734.842801     228500
4     218902.816808     200624
..              ...        ...
141   136618.296087     143000
142   145713.563889     143500
143   429878.200479     345000
144   209332.961672     246578
145   155690.060582     149500

[146 rows x 2 columns]
```

### Run Generated Code

```
cd outputs/
python final_script.py
```


