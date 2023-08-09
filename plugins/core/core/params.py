# Copyright 2023 The SapientML Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from pydantic import BaseModel, Field, validator
from sapientml.params import Config, Task
from sapientml.result import Code

from .meta_features import (
    MetaFeatures,
    generate_column_meta_features,
    generate_model_meta_features,
    generate_pp_meta_features,
)

PipelineSkeleton = dict[str, Union[float, dict[str, Union[float, list[str], list[dict[str, Union[float, int, str]]]]]]]

MAX_NUM_OF_COLUMNS = 10000000
MAX_COLUMN_NAME_LENGTH = 1000


class Column(BaseModel):
    dtype: str
    meta_features: Optional[MetaFeatures]
    has_negative_value: bool

    @validator("dtype")
    def check_dtype(cls, v):
        if len(v) > 100:
            raise ValueError(f"'{v}' is invalid as a dtype")
        return v

    @validator("meta_features")
    def check_meta_features(cls, v):
        if v is None:
            return v
        for k, _v in v.items():
            if not re.match(r"feature:[a-z_0-9]+", k):
                raise ValueError(f"'{k}' is invalid as a feature name")
            if isinstance(_v, int):
                pass
            elif isinstance(_v, float):
                pass
            elif isinstance(_v, str) and len(_v) >= MAX_COLUMN_NAME_LENGTH:
                raise ValueError(f"Meta feature string value must be shorter than {MAX_COLUMN_NAME_LENGTH}")
            elif isinstance(_v, list):
                for s in _v:
                    if not isinstance(s, str):
                        raise ValueError("The list must contain string values only")
                    elif len(s) >= MAX_COLUMN_NAME_LENGTH:
                        raise ValueError(
                            f"Column name length in meta features must be shorter than {MAX_COLUMN_NAME_LENGTH}"
                        )
        return v


class DatasetSummary(BaseModel):
    columns: dict[str, Column]
    meta_features_pp: MetaFeatures
    meta_features_m: MetaFeatures
    has_multi_class_targets: bool
    has_inf_value_targets: bool
    cols_almost_missing_string: Optional[list[str]] = None
    cols_almost_missing_numeric: Optional[list[str]] = None

    @validator("columns", "cols_almost_missing_string", "cols_almost_missing_numeric")
    def check_num_of_columns(cls, v):
        if v is None:
            return v
        if len(v.keys() if isinstance(v, dict) else v) >= MAX_NUM_OF_COLUMNS:
            raise ValueError(f"The number of columns must be smaller than {MAX_NUM_OF_COLUMNS}")
        return v

    @validator("columns", "cols_almost_missing_string", "cols_almost_missing_numeric")
    def check_column_name_length(cls, v):
        if v is None:
            return v
        for _v in v.keys() if isinstance(v, dict) else v:
            if len(_v) >= MAX_COLUMN_NAME_LENGTH:
                raise ValueError(f"Column name length must be shorter than {MAX_COLUMN_NAME_LENGTH}")
        return v

    @validator("meta_features_pp", "meta_features_m")
    def check_meta_features(cls, v):
        for k, _v in v.items():
            if not re.match(r"feature:[a-z_0-9]+", k):
                raise ValueError(f"'{k}' is invalid as a feature name")
            if isinstance(_v, int):
                pass
            elif isinstance(_v, float):
                pass
            elif isinstance(_v, str) and len(_v) >= MAX_COLUMN_NAME_LENGTH:
                raise ValueError(f"Meta feature string value must be shorter than {MAX_COLUMN_NAME_LENGTH}")
            elif isinstance(_v, list):
                for s in _v:
                    if not isinstance(s, str):
                        raise ValueError("The list must contain string values only")
                    elif len(s) >= MAX_COLUMN_NAME_LENGTH:
                        raise ValueError(
                            f"Column name length in meta features must be shorter than {MAX_COLUMN_NAME_LENGTH}"
                        )
        return v


class ModelLabel(BaseModel):
    label_name: str
    predict_proba: bool = False
    hyperparameters: Optional[Any] = None
    meta_features: list[Any] = Field(default_factory=list)

    def __str__(self):
        return self.label_name

    def __repr__(self):
        return str(self)


class Pipeline(Code):
    task: Task
    dataset_summary: DatasetSummary
    config: Config
    adaptation_metric: Optional[str] = None
    all_columns_datatypes: dict = Field(default_factory=dict)
    model: Optional[ModelLabel] = None
    inverse_target: bool = False
    sparse_matrix: bool = False  # Whether the data is converted to sparse matrix in the pipeline
    train_column_names: list[str] = Field(default_factory=list)
    test_column_names: list[str] = Field(default_factory=list)
    train_ignore_columns: list[str] = Field(default_factory=list)
    test_ignore_columns: list[str] = Field(default_factory=list)

    # To handle following case;
    #   metrics : Accuracy
    #   task    : multi-class and multi-targets
    # because sklearn.accuracy_score doesn't support multi-class
    is_multi_class_multi_targets: bool = False

    id_columns_for_prediction: list[str] = Field(default_factory=list)
    output_dir_path: str = ""

    # pipeline json
    pipeline_json: dict = Field(default_factory=lambda: defaultdict(dict))

    labels: Optional[PipelineSkeleton] = None


class CoreResult(BaseModel):
    pipelines: list[Pipeline]
    labels: PipelineSkeleton


def summarize_dataset(df_train: pd.DataFrame, task: Task) -> DatasetSummary:
    """
    Args:
        - df_train      : Input dataset
        - task          : Task

    Returns:
        DatasetSummary
    """
    is_multi_classes: list[bool] = []
    for target in task.target_columns:
        is_multi_classes.append(len(df_train[target].unique()) > 1)
    has_multi_class_targets = all(is_multi_classes)

    has_inf_value_targets = bool(df_train[task.target_columns].isin([np.inf, -np.inf]).sum().sum() > 0)

    # handle almost_missing columns
    cols_all_missing = df_train.columns[df_train.isnull().all()]
    cols_almost_missing = df_train.columns[df_train.isna().sum() / len(df_train) > 0.8]
    cols_almost_missing = sorted(list(set(cols_almost_missing) - set(cols_all_missing)))
    cols_almost_missing_numeric = [col for col in cols_almost_missing if is_numeric_dtype(df_train[col])]
    cols_almost_missing_string = [col for col in cols_almost_missing if not is_numeric_dtype(df_train[col])]

    columns = dict()
    for column_name in df_train.columns:
        if column_name not in task.ignore_columns:
            meta_features = generate_column_meta_features(df_train[[column_name]])
        else:  # in case ignore_columns (columns.keys() be called by Adaptation class)
            meta_features = None

        columns[column_name] = Column(
            dtype=str(df_train[column_name].dtype),
            meta_features=meta_features,
            has_negative_value=bool(
                is_numeric_dtype(df_train[column_name]) and (df_train[[column_name]].values <= 0).any()
            ),
        )

    # Generate the meta-features
    meta_features_pp = generate_pp_meta_features(df_train, task.target_columns)

    is_clf_task = 1 if task.task_type == "classification" else 0
    meta_features_m = generate_model_meta_features(df_train, task.target_columns, is_clf_task)

    return DatasetSummary(
        columns=columns,
        meta_features_pp=meta_features_pp,
        meta_features_m=meta_features_m,
        has_multi_class_targets=has_multi_class_targets,
        has_inf_value_targets=has_inf_value_targets,
        cols_almost_missing_string=cols_almost_missing_string,
        cols_almost_missing_numeric=cols_almost_missing_numeric,
    )
