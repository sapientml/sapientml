# SapientML

- [How to contribute this repository](./CONTRIBUTING.md)

# Getting Started
## Prerequisites

- Python >=3.9,<3.11

## Installation

```
pip install sapientml
```

### Generate Code in a notebook

Please download [housing-prices.csv](https://github.com/F-AutoML/sapientml/files/10430539/housing-prices.csv) to execute the following code.

```py
import pandas as pd

df = pd.read_csv("housing-prices.csv")

from sapientml import SapientML

sml = SapientML()

ret = sml.generate_code(
    training_data=df,
    task_type="regression",
    target_columns=["SalePrice"],
    ignore_columns=["Id"],
    adaptation_metric="RMSE",
    hyperparameter_tuning=False,
    id_columns_for_prediction=["Id"]
)
ret.save(
    "outputs",
    save_dev_scripts=True,
    save_user_scripts=True,
    save_datasets=True,
    save_running_arguments=True,
)
```

### Run Generated Code

```
cd outputs/
python final_script.py
```


