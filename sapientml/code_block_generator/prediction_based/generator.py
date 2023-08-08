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

import numpy as np
import pandas as pd
from sapientml.code_block_generator.rule_based.generator import check_cols_has_symbols, remove_symbols
from sapientml.params import Config, Task, _confirm_mixed_type
from sapientml.util.logging import setup_logger

from . import ps_macros
from .adaptation.generation.template_based_adaptation import Adaptation
from .params import CoreResult, summarize_dataset
from .seeding.predictor import predict

logger = setup_logger()


def generate_code_sapientml(df: pd.DataFrame, task: Task, config: Config, logger):
    # Apply column name changes(mixed_type__num, mixed_type__str, Remove special symbols) to avoid KeyError
    cols_has_symbols = []
    cols_has_symbols = check_cols_has_symbols(df.columns.to_list())
    if cols_has_symbols:
        df = df.rename(columns=lambda col: remove_symbols(col) if col in cols_has_symbols else col)
        task.target_columns = [remove_symbols(col) if col in cols_has_symbols else col for col in task.target_columns]
    cols_numeric_and_string = _confirm_mixed_type(df)
    for col in cols_numeric_and_string:
        df[col + "__str"] = np.where(pd.to_numeric(df[col], errors="coerce").isnull(), df[col], np.nan)
        df[col + "__num"] = np.where(pd.to_numeric(df[col], errors="coerce").isnull(), np.nan, df[col]).astype(float)
        df = df.drop(col, axis=1)
    # inf is converted to nan, so convert here to apply imputer to inf columns
    X = df.columns.drop(task.target_columns)
    df[X] = df[X].replace([np.inf, -np.inf], np.nan)

    dataset_summary = summarize_dataset(df, task)  # type: ignore
    if dataset_summary.has_inf_value_targets:
        raise ValueError("Stopped generation because target columns have infinity value.")

    # discard columns with analysis
    # NOTE: The following code modify task.ignore_columns because ignore_columns is the same instance as task.ignore_columns.
    # 1. columns marked as STR_OTHER
    if ps_macros.STR_OTHER in dataset_summary.meta_features_pp:
        undetermined_column_names = dataset_summary.meta_features_pp[ps_macros.STR_OTHER]
        if isinstance(undetermined_column_names, list):
            task.ignore_columns += undetermined_column_names
    del dataset_summary.meta_features_pp[ps_macros.STR_OTHER]
    # 2. columns with all null values
    if ps_macros.ALL_MISSING_PRESENCE in dataset_summary.meta_features_pp:
        column_names_with_all_missing_values = dataset_summary.meta_features_pp[ps_macros.ALL_MISSING_PRESENCE]
        if isinstance(column_names_with_all_missing_values, list):
            task.ignore_columns += column_names_with_all_missing_values
    del dataset_summary.meta_features_pp[ps_macros.ALL_MISSING_PRESENCE]

    labels = predict(task, dataset_summary)
    adapt = Adaptation(
        labels,
        task,
        dataset_summary,
        config,
    )
    pipelines = adapt.run_adaptation()
    result = CoreResult(pipelines=pipelines, labels=labels)

    return result
