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

from typing import List, Optional

import pandas as pd


class SapientMLSuggestion:
    def __init__(
        self,
        target_columns: List[str],
        dataframe: Optional[pd.DataFrame] = None,
    ):
        self.target_columns = target_columns
        self.dataframe = dataframe

    def suggest_task(self, th_val=30):
        if self.dataframe is None:
            raise ValueError("`dataframe` is not specified.")
        task_suggestion_dict = {}
        for target_column in self.target_columns:
            unique_num = self.dataframe[target_column][self.dataframe[target_column].notna()].nunique()
            data = self.dataframe[target_column][self.dataframe[target_column].notna()]
            ratio = unique_num / data.count()

            if self.dataframe[target_column].dtypes == "object":
                task_params = "classification"
            else:
                if ratio < 0.05 and unique_num < th_val:
                    task_params = "classification"
                else:
                    task_params = "regression"
            task_suggestion_dict[target_column] = task_params

        count_classification = list(task_suggestion_dict.values()).count("classification")
        count_regression = list(task_suggestion_dict.values()).count("regression")

        if count_classification > count_regression:
            task_suggestion = "classification"
        else:
            task_suggestion = "regression"

        return task_suggestion
