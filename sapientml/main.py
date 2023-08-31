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

# from msilib.schema import Error
from importlib.metadata import entry_points
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
from sapientml.suggestion import SapientMLSuggestion

from .executor import run
from .macros import Metric
from .params import Dataset, Task, save_file
from .util.logging import setup_logger

logger = setup_logger()

DEFAULT_OUTPUT_DIR = "./outputs"


def _check_stratification(
    full_dataset,
    target_columns,
    task_type,
    adaptation_metric,
    split_stratification,
    stratify_threshold,
):
    base_threshold = 0.2
    temp_res = full_dataset[target_columns[0]].value_counts()
    min_value = temp_res.min()
    min_value_ratio = min_value / temp_res.sum()
    ratio_threshold = base_threshold / len(temp_res)
    if task_type == "classification":
        if min_value < stratify_threshold:
            if adaptation_metric == "LogLoss":
                raise ValueError(
                    f"Target value {temp_res.idxmin()} appears only {min_value} times, the input data cannot be split for LogLoss metric. Please review the target column or set another adaptation_metric"
                )
            elif split_stratification:
                raise ValueError(
                    f"split_stratification=True, but the input data cannot be split using stratification because target value {temp_res.idxmin()} appears only {min_value} times. Please set split_stratification as False"
                )
            else:
                return False
        elif min_value_ratio < ratio_threshold:
            if split_stratification is None:
                return True
            elif not split_stratification:
                logger.warning(
                    "split_stratification is set False, but setting True is recommended because the target is imbalance."
                )
            return split_stratification
        else:
            if split_stratification is None:
                return False
            return split_stratification
    else:
        return False


class SapientML:
    def __init__(
        self,
        # Dataset
        target_columns: list[str],
        # Task
        task_type: Optional[Literal["classification", "regression"]] = None,
        adaptation_metric: Optional[str] = None,
        split_method: Literal["random", "time", "group"] = "random",
        split_seed: int = 17,
        split_train_size: float = 0.75,
        split_column_name: Optional[str] = None,
        time_split_num: int = 5,
        time_split_index: int = 4,
        split_stratification: Optional[bool] = None,
        model_type: str = "sapientml",
        **kwargs,
    ):
        """
        Generate ML scripts for input data.

        Parameters
        ----------
        target_columns: list[str]
            Names of target columns
        task_type: 'classification' or 'regression'
            Specify classification or regression.
        adaptation_metric: str
            Metric for evaluation.
            Classification: 'f1', 'auc', 'ROC_AUC', 'accuracy', 'Gini', 'LogLoss',
            'MCC'(Matthews correlation coefficient), 'QWK'(Quadratic weighted kappa).
            Regression: 'r2', 'RMSLE', 'RMSE', 'MAE'.
        split_method: 'random', 'time', or 'group'
            Method of train-test split.
            'random' uses random split.
            'time' requires 'split_column_name'.
            This sorts the data rows based on the column, and then splits data.
            'group' requires 'split_column_name'.
            This split the data so as not to split rows with the same value of 'split_column_name'
            into train and test data.
            Currently, this option is not valid in the hyperparameter tuning.
            Don't set time or group when hyperparameter_tuning=True.
        split_seed: int
            Random seed for train-test split.
            Ignored when split_method='time'.
        split_train_size: float
            The ratio of training size to input data.
            Ignored when split_method='time'.
        split_column_name: str
            Name of the column used to split.
            Ignored when split_method='random'
        time_split_num: int
            Passed to TimeSeriesSplit's n_splits.
            Valid only when split_method='time'.
        time_split_index: int
            The index of the split from TimeSeriesSplit.
            Valid only when split_method='time'.
        split_stratification: bool
            To perform stratification in train-test split.

        """

        self.task = Task(
            target_columns=target_columns,
            task_type=task_type,
            split_method=split_method,
            split_seed=split_seed,
            split_train_size=split_train_size,
            split_column_name=split_column_name,
            time_split_num=time_split_num,
            time_split_index=time_split_index,
            adaptation_metric=adaptation_metric,
            split_stratification=split_stratification,
        )
        eps_generator = entry_points(group="sapientml.pipeline_generator")
        eps_config = entry_points(group="sapientml.config")
        if eps_generator[model_type] and eps_config[model_type]:
            self._Generator = eps_generator[model_type].load()
            self._Config = eps_config[model_type].load()
        else:
            raise ValueError(f"Model '{model_type}' is invalid.")

        self.generator = self._Generator(**kwargs)
        self.config = self.generator.config
        self.config.postinit()

    def fit(
        self,
        training_data: Union[pd.DataFrame, str],
        validation_data: Optional[Union[pd.DataFrame, str]] = None,
        test_data: Optional[Union[pd.DataFrame, str]] = None,
        save_datasets_format: Literal["csv", "pickle"] = "pickle",
        csv_encoding: Literal["UTF-8", "SJIS"] = "UTF-8",
        csv_delimiter: str = ",",
        ignore_columns: Optional[list[str]] = None,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        codegen_only: bool = False,
    ):
        """
        Generate ML scripts for input data.

        Parameters
        ----------
        training_data: pandas.DataFrame or str
            Training dataframe.
            When str, this is regarded as a file path.
        validation_data: pandas.DataFrame, str or None
            Validation dataframe.
            When str, this is regarded as file paths.
            When None, validation data is extracted from training data by split.
        test_data: pandas.DataFrame, str, or None
            Test dataframes.
            When str, they are regarded as file paths.
            When None, test data is extracted from training data by split.
        save_datasets_format: 'csv' or 'pickle'
            Data format when the input dataframes are written to files.
            Ignored when all inputs are specified as file path.
        csv_encoding: 'UTF-8' or 'SJIS'
            Encoding method when csv files are involved.
            Ignored when only pickle files are involved.
        csv_delimiter: str
            Delimiter to read csv files
        ignore_columns: list[str]
            Column names which must not be used and must be dropped.
        output_dir: str
            Output dir
        codegen_only: bool
            Generated code is not saved if True

        Returns
        -------
        """

        if ignore_columns is None:
            ignore_columns = []

        logger.info("Loading dataset...")

        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = Dataset(
            training_data=training_data,
            validation_data=validation_data,
            test_data=test_data,
            csv_encoding=csv_encoding,
            csv_delimiter=csv_delimiter,
            save_datasets_format=save_datasets_format,
            ignore_columns=ignore_columns,
            output_dir=self.output_dir,
        )
        self.dataset.check_dataframes(self.task.target_columns)

        if self.task.task_type is None:
            self.task.task_type = SapientMLSuggestion(
                self.task.target_columns, self.dataset.training_dataframe
            ).suggest_task()

        if self.task.adaptation_metric is None and self.task.task_type:
            if self.task.task_type == "regression":
                logger.warning("Metric is not specified. Use 'r2' by default.")
            else:
                logger.warning("Metric is not specified. Use 'f1' by default.")
            self.task.adaptation_metric = Metric.get_default_value(self.task.task_type)

        self.task.adaptation_metric = Metric.get(self.task.adaptation_metric)

        if not Metric.metric_match_task_type(self.task.adaptation_metric, self.task.task_type):
            logger.warning(f"{self.task.adaptation_metric} is not a metric for {self.task.task_type}")

        if self.task.task_type == "classification":
            is_multioutput = False
            if len(self.task.target_columns) > 1:
                is_multioutput = True
            for column in self.task.target_columns:
                if isinstance(training_data, pd.DataFrame) and len(training_data[column].unique()) > 2:
                    self.task.is_multiclass = True

            if is_multioutput:
                if self.task.is_multiclass and not Metric.metric_support_multiclass_multioutput(
                    self.task.adaptation_metric
                ):
                    logger.warning(
                        f"{self.task.adaptation_metric} does not support Multiclass-Multioutput classification. Execution of candidate script raises an exception."
                    )
                elif not Metric.metric_support_multioutput(self.task.adaptation_metric):
                    logger.warning(
                        f"{self.task.adaptation_metric} does not support Multioutput (Multilabel) classification. Execution of candidate script raises an exception."
                    )

        training_dataframe, validation_dataframe, test_dataframe = self.dataset.get_dataframes()
        stratify_threshold = 3
        if validation_dataframe is not None:
            stratify_threshold -= 1
        if test_dataframe is not None:
            stratify_threshold -= 1
        if len(self.task.target_columns) == 1:
            self.task.split_stratification = _check_stratification(
                training_dataframe,
                self.task.target_columns,
                self.task.task_type,
                self.task.adaptation_metric,
                self.task.split_stratification,
                stratify_threshold,
            )
        elif self.task.task_type == "classification" and self.task.split_stratification:
            raise ValueError("Stratification for multiple target columns is not supported.")

        self.generator.generate_pipeline(self.dataset, self.task)
        self.generator.save(self.output_dir)

        if codegen_only:
            return

        logger.info("Building model by generated pipeline...")
        run(str(self.output_dir / "final_train.py"), self.config.timeout_for_test)
        logger.info("Done.")

    def predict(
        self,
        test_data: Union[pd.DataFrame, str],
    ):
        if isinstance(test_data, pd.DataFrame):
            filename = "test." + "pkl" if self.dataset.save_datasets_format == "pickle" else "csv"
            save_file(test_data, self.output_dir / filename, self.dataset.csv_encoding, self.dataset.csv_delimiter)
        else:
            return

        logger.info("Predicting by built model...")
        run(str(self.output_dir / "final_predict.py"), self.config.timeout_for_test)
        result = pd.read_csv(self.output_dir / "prediction_result.csv")
        return result

    def get_config(self):
        return self.config
