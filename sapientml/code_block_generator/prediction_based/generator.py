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

from pathlib import Path

import pandas as pd
from sapientml.params import Config, Task
from sapientml.util.logging import setup_logger

from . import ps_macros
from .adaptation.generation.template_based_adaptation import Adaptation
from .params import CoreResult, summarize_dataset
from .seeding.predictor import predict

model_dir_path_default = Path(__file__).parent / "models"

logger = setup_logger()


def generate_code_sapientml(df: pd.DataFrame, task: Task, config: Config, logger):
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
