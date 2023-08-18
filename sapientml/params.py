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

import os
import warnings
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, validator

from .macros import Metric
from .util.logging import setup_logger

MAX_NUM_OF_COLUMNS = 10000000
MAX_PATH_LENGTH = 1000
MAX_SEED = 2**32 - 1
MAX_COLUMN_NAME_LENGTH = 1000
MAX_TIME_SPLIT_NUM = 10000
MAX_TIME_SPLIT_INDEX = 10000
MAX_N_MODELS = 30
MAX_HPO_N_TRIALS = 100000
MAX_HPO_TIMEOUT = 500000
DEFAULT_OUTPUT_DIR = "."

logger = setup_logger()


def _read_file(filepath: str, csv_encoding: str, csv_delimiter: str) -> pd.DataFrame:
    if filepath.endswith(".pkl"):
        res_df = pd.read_pickle(filepath)
    else:
        res_df = pd.read_csv(filepath, encoding=csv_encoding, delimiter=csv_delimiter)
    return res_df


def _is_strnum_column(c):
    c2 = c.loc[c.notnull()]
    c2 = pd.to_numeric(c2, errors="coerce")
    ratio = c2.notnull().sum() / c2.shape[0]
    return ratio > 0.9


def _confirm_mixed_type(df: pd.DataFrame) -> list[str]:
    mixed_df_cols = []
    for df_col in df.columns:
        df_per_col = df[df_col].replace([-np.inf, np.inf], np.nan)
        df_per_col2 = df_per_col[df_per_col.notnull()]
        types_per_col = pd.api.types.infer_dtype(df_per_col2)

        is_strnum = _is_strnum_column(df[df_col])

        if types_per_col == "mixed" or types_per_col == "mixed-integer" or (types_per_col == "string" and is_strnum):
            mixed_df_cols.append(df_col)

    return mixed_df_cols


def _is_date_column(c):
    c2 = c.loc[c.notnull()]
    if c2.shape[0] > 1000:
        c2 = c2.sample(1000, random_state=17)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c2 = pd.to_datetime(c2, errors="coerce")
    ratio = c2.notnull().sum() / c2.shape[0]
    return ratio > 0.8


class CancellationToken:
    isTriggered = False


class Code(BaseModel):
    validation: str = ""
    test: str = ""
    train: str = ""
    predict: str = ""

    def __add__(self, other):
        return Code(
            validation=self.validation + other.validation,
            test=self.test + other.test,
            train=self.train + other.train,
            predict=self.predict + other.predict,
        )


class RunningResult(BaseModel):
    output: str
    error: str
    returncode: int
    time: int


class PipelineResult(BaseModel):
    score: Optional[float]
    metric: Optional[str]
    best_params: Optional[dict]


class Task(BaseModel):
    target_columns: list[str]
    task_type: Literal["classification", "regression"]
    ignore_columns: list[str]
    split_method: Literal["random", "time", "group"]
    split_seed: int
    split_train_size: float
    split_column_name: Optional[str]
    time_split_num: int
    time_split_index: int
    adaptation_metric: str
    n_models: int
    seed_for_model: int
    id_columns_for_prediction: list[str]
    use_word_list: Optional[Union[list[str], dict[str, list[str]]]] = None
    use_pos_list: Optional[list[str]] = None
    use_word_stemming: Optional[bool] = None
    split_stratify: Optional[bool] = None
    is_multiclass: Optional[bool] = None

    @validator(
        "target_columns",
        "ignore_columns",
        "id_columns_for_prediction",
        "use_word_list",
        "use_pos_list",
    )
    def check_num_of_column_names(cls, v):
        if v is None:
            return v
        if len(v.keys() if isinstance(v, dict) else v) >= MAX_NUM_OF_COLUMNS:
            raise ValueError(f"The number of columns must be smaller than {MAX_NUM_OF_COLUMNS}")
        return v

    @validator(
        "target_columns",
        "ignore_columns",
        "id_columns_for_prediction",
        "use_word_list",
        "use_pos_list",
    )
    def check_column_name_length(cls, v):
        if v is None:
            return v
        for _v in v.keys() if isinstance(v, dict) else v:
            if len(_v) >= MAX_COLUMN_NAME_LENGTH:
                raise ValueError(f"Column name length must be shorter than {MAX_COLUMN_NAME_LENGTH}")
        return v

    @validator("split_seed", "seed_for_model")
    def check_seed(cls, v):
        if v < 0 or MAX_SEED < v:
            raise ValueError(f"{v} is out of [0, {MAX_SEED}]")
        return v

    @validator("split_train_size")
    def check_split_train_size(cls, v):
        if v <= 0 or 1 <= v:
            raise ValueError(f"{v} is out of (0, 1)")
        return v

    @validator("split_column_name")
    def check_split_column_name(cls, v):
        if v is not None and len(v) >= MAX_COLUMN_NAME_LENGTH:
            raise ValueError(f"Column name length must be shorter than {MAX_COLUMN_NAME_LENGTH}")
        return v

    @validator("time_split_num")
    def check_time_split_num(cls, v):
        if v < 1 or MAX_TIME_SPLIT_NUM < v:
            raise ValueError(f"{v} is out of [1, {MAX_TIME_SPLIT_NUM}]")
        return v

    @validator("time_split_index")
    def check_time_split_index(cls, v):
        if v < 0 or MAX_TIME_SPLIT_INDEX < v:
            raise ValueError(f"{v} is out of [1, {MAX_TIME_SPLIT_INDEX}].")
        return v

    @validator("adaptation_metric")
    def check_metric(cls, v):
        try:
            Metric.get(v)
        except NotImplementedError:
            raise ValueError(f"'{v}' is invalid as a metric.")
        return v

    @validator("n_models")
    def check_n_models(cls, v):
        if v <= 0 or MAX_N_MODELS < v:
            raise ValueError(f"{v} is out of [1, {MAX_N_MODELS}]")
        return v


class Config(BaseModel):
    use_hyperparameters: bool = False
    impute_all: bool = True
    hyperparameter_tuning: bool = False
    hyperparameter_tuning_n_trials: int = 10
    hyperparameter_tuning_timeout: int = 0
    hyperparameter_tuning_random_state: int = 1023
    predict_option: Literal["default", "probability"] = "default"
    permutation_importance: bool = True

    @validator("hyperparameter_tuning_n_trials")
    def check_hyperparameter_tuning_n_trials(cls, v):
        if v < 1 or MAX_HPO_N_TRIALS < v:
            raise ValueError(f"{v} is out of [1, {MAX_HPO_N_TRIALS}]")
        return v

    @validator("hyperparameter_tuning_timeout")
    def check_hyperparameter_tuning_timeout(cls, v):
        if v < 0 or MAX_HPO_TIMEOUT < v:
            raise ValueError(f"{v} is out of [0, {MAX_HPO_TIMEOUT}]")
        return v

    @validator("hyperparameter_tuning_random_state")
    def check_hyperparameter_tuning_random_state(cls, v):
        if v < 0 or MAX_SEED < v:
            raise ValueError(f"{v} is out of [0, {MAX_SEED}]")
        return v


class Dataset:
    def __init__(
        self,
        training_data: Union[pd.DataFrame, str],
        validation_data: Union[pd.DataFrame, str, None] = None,
        test_data: Union[pd.DataFrame, str, None] = None,
        csv_encoding: Literal["UTF-8", "SJIS"] = "UTF-8",
        csv_delimiter: str = ",",
        save_datasets_format: Literal["csv", "pickle"] = "pickle",
        ignore_columns: Optional[List[str]] = None,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        self.ignore_columns = [] if ignore_columns is None else ignore_columns
        self.csv_encoding = csv_encoding
        self.csv_delimiter = csv_delimiter
        self.save_datasets_format = save_datasets_format

        if isinstance(training_data, str):
            self.training_dataframe = _read_file(training_data, csv_encoding, csv_delimiter)
            self.training_data_path = training_data
        else:
            self.training_dataframe = training_data.copy()
            if save_datasets_format == "pickle":
                self.training_data_path = os.path.join(output_dir, "training.pkl")
                self.training_dataframe.to_pickle(self.training_data_path)
            else:
                self.training_data_path = os.path.join(output_dir, "training.csv")
                self.training_dataframe.to_csv(
                    self.training_data_path, encoding=csv_encoding, sep=csv_delimiter, index=False
                )

        # NOTE: self.validation_data and self.test_data can be None
        if validation_data is not None and test_data is None:
            raise ValueError(
                "test_data must not be None when validation_data is specified. test_data should be specified instead of validation_data."
            )
        if isinstance(validation_data, str):
            self.validation_dataframe = _read_file(validation_data, csv_encoding, csv_delimiter)
            self.validation_data_path = validation_data
        elif isinstance(validation_data, pd.DataFrame):
            self.validation_dataframe = validation_data.copy()
            if save_datasets_format == "pickle":
                self.validation_data_path = os.path.join(output_dir, "validation.pkl")
                self.validation_dataframe.to_pickle(self.validation_data_path)
            else:
                self.validation_data_path = os.path.join(output_dir, "validation.csv")
                self.validation_dataframe.to_csv(
                    self.validation_data_path, encoding=csv_encoding, sep=csv_delimiter, index=False
                )
        else:
            self.validation_dataframe = None
            self.validation_data_path = None

        if isinstance(test_data, str):
            self.test_dataframe = _read_file(test_data, csv_encoding, csv_delimiter)
            self.test_data_path = test_data
        elif isinstance(test_data, pd.DataFrame):
            self.test_dataframe = test_data.copy()
            if save_datasets_format == "pickle":
                self.test_data_path = os.path.join(output_dir, "test.pkl")
                self.test_dataframe.to_pickle(self.test_data_path)
            else:
                self.test_data_path = os.path.join(output_dir, "test.csv")
                self.test_dataframe.to_csv(self.test_data_path, encoding=csv_encoding, sep=csv_delimiter, index=False)
        else:
            self.test_dataframe = None
            self.test_data_path = None

    def check_dataframes(
        self,
        target_columns: List[str],
    ):
        # 1. Check status of each dataframe
        self._check_single_dataframe(self.training_dataframe, target_columns, "train")
        if self.validation_dataframe is not None:
            self._check_single_dataframe(self.validation_dataframe, target_columns, "validation")
        if self.test_dataframe is not None:
            self._check_single_dataframe(self.test_dataframe, target_columns, "test")

        def format_warning(inconsistent_df_cols, target_data_name1, target_data_name2):
            for inconsistent_df_col in inconsistent_df_cols:
                logger.warning(
                    f"types of {inconsistent_df_col} are inconsistent between {target_data_name1} and {target_data_name2} dataframes."
                )

        # 2. Check whether column names and orders are consistency between two dataframes
        if self.validation_dataframe is not None:
            if not self._is_same(
                self.training_dataframe.drop(self.ignore_columns, axis=1, errors="ignore"),
                self.validation_dataframe.drop(self.ignore_columns, axis=1, errors="ignore"),
            ):
                raise ValueError(
                    "Train/validation dataframes have different column names or its order. It must be same among the dataframes."
                )
        if self.test_dataframe is not None:
            if not self._is_same(
                self.training_dataframe.drop(target_columns, axis=1).drop(self.ignore_columns, axis=1, errors="ignore"),
                self.test_dataframe.drop(target_columns, axis=1, errors="ignore").drop(
                    self.ignore_columns, axis=1, errors="ignore"
                ),
            ):
                raise ValueError(
                    "Train/test dataframes have different column names or its order. It must be same among the dataframes."
                )

        # 3. Check whether types are consistency between two dataframes
        if self.validation_dataframe is not None:
            format_warning(
                self._confirm_consistent_type(self.training_dataframe, self.validation_dataframe),
                "train",
                "validation",
            )
        if self.test_dataframe is not None:
            format_warning(
                self._confirm_consistent_type(self.training_dataframe, self.test_dataframe),
                "train",
                "test",
            )

    def _check_single_dataframe(self, df: pd.DataFrame, target_columns: List[str], target_data_name: str) -> bool:
        # 0. Target existence
        # test data doesn't have to have target columns because there is the case where only prediction is needed.
        if (target_data_name != "test") and (not set(target_columns).issubset(set(df.columns))):
            raise ValueError(f"{target_data_name} doesn't have target columns")

        # 1. Check NaN in target column
        if set(target_columns).issubset(set(df.columns)) and (not self._isnot_nan(df, target_columns)):
            raise ValueError(f"target column of {target_data_name} dataframe has NaN or Inf.")

        # 2. Check whether index names are unique
        if not self._is_index_unique(df):
            raise ValueError(f"Index names of {target_data_name} dataframe are not unique.")

        # 3. Check whether column names are unique
        if not self._is_columns_unique(df):
            logger.warning(f"Column names of {target_data_name} dataframe are not unique.")

        # 4. Check strnum mixed type
        mixed_df_cols = _confirm_mixed_type(df)
        for mixed_df_col in mixed_df_cols:
            logger.warning(f"{mixed_df_col} would have mixed type in {target_data_name} dataframe.")

        # 5. Check whether datetime columns are correct value
        incorrect_info = self._confirm_date_type(df)
        incorrect_cols = list(incorrect_info.keys())
        incorrect_summary = []
        num_shown_for_one_column = 3
        for col, dt_column in incorrect_info.items():
            for index, value in dt_column.iloc[:num_shown_for_one_column].items():
                incorrect_summary.append((col, index, value))
        if incorrect_summary:
            logger.warning(
                f"The follwing columns would have incorrect datetime values in {target_data_name} dataframe: {incorrect_cols}"
            )
            logger.warning(f"For example, (column_name, index, value) = {incorrect_summary}")

        return True

    def _isnot_nan(self, df: pd.DataFrame, target_columns: list[str]) -> bool:
        for target_column in target_columns:
            target_data = df[target_column].replace([-np.inf, np.inf], np.nan)
            if target_data.isna().any():
                return False
        return True

    def _is_columns_unique(self, df: pd.DataFrame) -> bool:
        if df.columns.is_unique:
            return True
        return False

    def _is_index_unique(self, df: pd.DataFrame) -> bool:
        if df.index.is_unique:
            return True
        return False

    def _confirm_consistent_type(self, df1: pd.DataFrame, df2: pd.DataFrame) -> list[str]:
        inconsistent_df_cols = []
        for df1_col in df1.columns:
            df1_col_type = pd.api.types.infer_dtype(df1[df1_col], skipna=True)
            if df1_col not in df2:
                continue
            df2_col_type = pd.api.types.infer_dtype(df2[df1_col], skipna=True)

            if df1_col_type != df2_col_type:
                inconsistent_df_cols.append(df1_col)

        return inconsistent_df_cols

    def _confirm_date_type(self, df: pd.DataFrame) -> dict:
        incorrect_info = {}
        # If only _is_date_column, all integer columns can be date columns.
        object_columns = df.select_dtypes("object").columns
        date_col = [col for col in object_columns if _is_date_column(df[col])]
        for col in date_col:
            org_column = df.loc[df[col].notnull(), col]
            dt_column = pd.to_datetime(org_column, errors="coerce")
            incorrect_values = org_column.loc[dt_column.isnull()]
            if incorrect_values.shape[0] > 0:
                incorrect_info[col] = incorrect_values
        return incorrect_info

    def _is_same(self, df1, df2):
        df1_columns = df1.columns
        df2_columns = df2.columns

        # The length of columns must be same
        if len(df1_columns) != len(df2_columns):
            return False

        # The item and order of columns must be same
        for i in range(len(df1_columns)):
            if df1_columns[i] != df2_columns[i]:
                return False
        return True

    def get_dataframes(self) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        return (self.training_dataframe, self.validation_dataframe, self.test_dataframe)
