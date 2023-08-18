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

import logging

# from msilib.schema import Error
import tempfile
from importlib.metadata import entry_points
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

from . import macros
from .executor import PipelineExecutor
from .params import CancellationToken, Config, Dataset, Task
from .result import SapientMLGeneratorResult
from .util.logging import setup_logger

INITIAL_TIMEOUT = 600


class SapientML:
    def __init__(
        self,
        loglevel=logging.INFO,
    ):
        self._logger = setup_logger()
        self._logger.setLevel(loglevel)

    def check_stratification(
        self,
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
                    self._logger.warning(
                        "split_stratification is set False, but setting True is recommended because the target is imbalance."
                    )
                return split_stratification
            else:
                if split_stratification is None:
                    return False
                return split_stratification
        else:
            return False

    def fit():
        pass

    def predict():
        pass

    def generate_code(
        self,
        # Dataset
        target_columns: list[str],
        training_data: Union[pd.DataFrame, str],
        validation_data: Optional[Union[pd.DataFrame, str]] = None,
        test_data: Optional[Union[pd.DataFrame, str]] = None,
        save_datasets_format: Literal["csv", "pickle"] = "pickle",
        csv_encoding: Literal["UTF-8", "SJIS"] = "UTF-8",
        csv_delimiter: str = ",",
        ignore_columns: Optional[list[str]] = None,
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
        # SharedConfig
        initial_timeout: Optional[int] = None,
        timeout_for_test: int = 0,
        cancel: Optional[CancellationToken] = None,
        # Config for generators
        # 1. rule_based
        use_word_list: Optional[Union[list[str], dict[str, list[str]]]] = None,
        use_pos_list: Optional[list[str]] = ["名詞", "動詞", "助動詞", "形容詞", "副詞"],
        use_word_stemming: bool = True,
        # 2. prediction_based
        # 2.1 model
        n_models: int = 3,
        model_seed: int = 551,
        use_hyperparameters: bool = False,
        hyperparameter_tuning: bool = False,
        hyperparameter_tuning_n_trials: int = 10,
        hyperparameter_tuning_timeout: Optional[int] = None,
        hyperparameter_tuning_random_state: int = 1021,
        # 2.2 postprocess
        predict_option: Literal["default", "probability"] = "default",
        permutation_importance: bool = True,
        id_columns_for_prediction: Optional[list[str]] = None,
    ) -> SapientMLGeneratorResult:
        """
        Generate ML scripts for input data.

        Parameters
        ----------
        target_columns: list[str]
            Names of target columns
        task_type: 'classification' or 'regression'
            Specify classification or regression.
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
        ignore_columns: list[str]
            Column names which must not be used and must be dropped.
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
        n_models: int
            Number of output models to file.
        model_seed: int
            Random seed for models such as RandomForestClassifier.
        adaptation_metric: str
            Metric for evaluation.
            Classification: 'f1', 'auc', 'ROC_AUC', 'accuracy', 'Gini', 'LogLoss',
            'MCC'(Matthews correlation coefficient), 'QWK'(Quadratic weighted kappa).
            Regression: 'r2', 'RMSLE', 'RMSE', 'MAE'.
        csv_delimiter: str
            Delimiter to read csv files
        initial_timeout: int
            Timelimit to execute each generated script.
        timeout_for_test: int
            Timelimit to execute test script (final_script) and Visualization.
        eda: bool
        visualization: bool
            On/Off of EDA and visualization.
            Ignored when add_explain=False in the save() function.
        permutation_importance: bool
            On/Off of permutation_importance.
            Ignored when add_explain=True in the save() function.
        use_hyperparameters: bool
            Specify whether or not hyperparameters are used.
        predict_option: 'default' or 'probability'
            Specify predict method (default: predict(), probability: predict_proba().)
        hyperparameter_tuning: bool
            On/Off of hyperparameter tuning.
        hyperparameter_tuning_n_trials: int
            The number of trials of hyperparameter tuning.
        hyperparameter_tuning_timeout: int
            Time limit for hyperparameter tuning in each generated script.
        hyperparameter_tuning_random_state: int
            Random seed for hyperparameter tuning.
        id_columns_for_prediction: list[str]
            Specify non-target column names for prediction.
        cancel: CancellationToken
            Object to interrupt evaluations.
        use_word_list: list[str] or dict[str]
            List of words to be used as features when generating explanatory variables from text.
            If dict type is specified, key must be a column name and value must be a list of words.
        use_pos_list: list[str]
            List of parts-of-speech to be used during text analysis.
            This variable is used for japanese texts analysis.
            Select the part of speech below.
            "名詞", "動詞", "形容詞", "形容動詞", "副詞", "接頭詞", "接続詞", "助詞", "助動詞", "感動詞", "連体詞", "記号", "フィラー"
        use_stemming: bool
            Specify whether or not word stemming is used.
            This variable is used for japanese texts analysis.
        split_stratification: bool
            To perform stratification in train-test split.

        Returns
        -------
        SapientMLGeneratorResult
            SapientMLGeneratorResult.save() must be called to save the generated scripts.
        """
        if task_type not in ["classification", "regression"]:
            raise ValueError(f"task_type '{task_type}' is invalid.")

        if len(target_columns) == 0:
            raise ValueError("Target columns are empty.")

        if ignore_columns is None:
            ignore_columns = []

        if id_columns_for_prediction is None:
            id_columns_for_prediction = []

        if initial_timeout is None:
            if hyperparameter_tuning:
                initial_timeout = 0
            else:
                initial_timeout = INITIAL_TIMEOUT
            if hyperparameter_tuning_timeout is None:
                hyperparameter_tuning_timeout = INITIAL_TIMEOUT
        elif hyperparameter_tuning_timeout is None:
            if hyperparameter_tuning:
                hyperparameter_tuning_timeout = initial_timeout
            else:
                hyperparameter_tuning_timeout = INITIAL_TIMEOUT

        if use_pos_list is None:
            use_pos_list = []

        if adaptation_metric is None:
            if task_type == "regression":
                self._logger.warning("Metric is not specified. Use 'r2' by default.")
            else:
                self._logger.warning("Metric is not specified. Use 'f1' by default.")
            adaptation_metric = macros.Metric.get_default_value(task_type)

        self._logger.info("Loading dataset...")

        with tempfile.TemporaryDirectory() as tmpdir_path_str:
            tmpdir = Path(tmpdir_path_str).absolute()
            tmpdir.mkdir(exist_ok=True)

            dataset = Dataset(
                training_data=training_data,
                validation_data=validation_data,
                test_data=test_data,
                csv_encoding=csv_encoding,
                csv_delimiter=csv_delimiter,
                save_datasets_format=save_datasets_format,
                ignore_columns=ignore_columns,
                output_dir=str(tmpdir),
                logger=self._logger,
            )
            dataset.check_dataframes(target_columns)

            training_dataframe, validation_dataframe, test_dataframe = dataset.get_dataframes()
            stratify_threshold = 3
            if validation_dataframe is not None:
                stratify_threshold -= 1
            if test_dataframe is not None:
                stratify_threshold -= 1
            split_stratify = False
            if len(target_columns) == 1:
                split_stratify = self.check_stratification(
                    training_dataframe,
                    target_columns,
                    task_type,
                    adaptation_metric,
                    split_stratification,
                    stratify_threshold,
                )
            elif task_type == "classification" and split_stratification:
                raise ValueError("Stratification for multiple target columns is not supported.")

            # Generate the meta-features
            self._logger.info("Generating meta features ...")
            task = Task(
                target_columns=target_columns,
                task_type=task_type,
                ignore_columns=ignore_columns,
                split_method=split_method,
                split_seed=split_seed,
                split_train_size=split_train_size,
                split_column_name=split_column_name,
                time_split_num=time_split_num,
                time_split_index=time_split_index,
                adaptation_metric=adaptation_metric,
                n_models=n_models,
                seed_for_model=model_seed,
                id_columns_for_prediction=id_columns_for_prediction,
                use_word_list=use_word_list,
                use_pos_list=use_pos_list,
                use_word_stemming=use_word_stemming,
                split_stratify=split_stratify,
            )

            adaptation_metric = macros.Metric.get(adaptation_metric)

            if not macros.Metric.metric_match_task_type(adaptation_metric, task_type):
                self._logger.warning(f"{adaptation_metric} is not a metric for {task_type}")

            if task_type == "classification":
                is_multiclass = False
                is_multioutput = False
                if len(target_columns) > 1:
                    is_multioutput = True
                for column in target_columns:
                    if len(training_data[column].unique()) > 2:
                        is_multiclass = True
                task.is_multiclass = is_multiclass

                if is_multioutput:
                    if is_multiclass and not macros.Metric.metric_support_multiclass_multioutput(adaptation_metric):
                        self._logger.warning(
                            f"{adaptation_metric} does not support Multiclass-Multioutput classification. Execution of candidate script raises an exception."
                        )
                    elif not macros.Metric.metric_support_multioutput(adaptation_metric):
                        self._logger.warning(
                            f"{adaptation_metric} does not support Multioutput (Multilabel) classification. Execution of candidate script raises an exception."
                        )

            config_instance = Config(
                use_hyperparameters=use_hyperparameters,
                impute_all=True,
                hyperparameter_tuning=hyperparameter_tuning,
                hyperparameter_tuning_n_trials=hyperparameter_tuning_n_trials,
                hyperparameter_tuning_timeout=hyperparameter_tuning_timeout,
                hyperparameter_tuning_random_state=hyperparameter_tuning_random_state,
                predict_option=predict_option,
                permutation_importance=permutation_importance,
            )

            self.dataset = dataset
            self.task = task
            self.config = config_instance
            eps = entry_points(group="pipeline_generator")
            self._generator = eps["sapientml_core"].load()(self.config)
            pipelines = self._generator.generate_pipeline(dataset, task)

            self._logger.info("Executing generated pipelines ...")
            executor = PipelineExecutor()
            pipeline_results = executor.execute(
                pipelines,
                initial_timeout,
                tmpdir,
                cancel,
            )

            lower_is_better = self.task.adaptation_metric in macros.metric_lower_is_better
            self._generator.evaluate(pipeline_results, lower_is_better)
            self._logger.info("Done.")

            final_script, candidate_scripts = self._generator.get_result()

            self.result = SapientMLGeneratorResult(
                final_script=final_script,
                candidate_scripts=candidate_scripts,
                training_data=training_data,
                validation_data=validation_data,
                test_data=test_data,
                training_data_path=dataset.training_data_path,
                validation_data_path=dataset.validation_data_path,
                test_data_path=dataset.test_data_path,
                csv_encoding=csv_encoding,
                tmpdir_path=tmpdir,
                target_columns=target_columns,
                task_type=task_type,
                ignore_columns=dataset.ignore_columns,
                split_method=split_method,
                split_seed=split_seed,
                split_train_size=split_train_size,
                split_column_name=split_column_name,
                time_split_num=time_split_num,
                time_split_index=time_split_index,
                n_models=n_models,
                seed_for_model=model_seed,
                adaptation_metric=adaptation_metric,
                csv_delimiter=csv_delimiter,
                initial_timeout=initial_timeout,
                timeout_for_test=timeout_for_test,
                use_hyperparameters=use_hyperparameters,
                predict_option=predict_option,
                hyperparameter_tuning=hyperparameter_tuning,
                hyperparameter_tuning_n_trials=hyperparameter_tuning_n_trials,
                hyperparameter_tuning_timeout=hyperparameter_tuning_timeout,
                hyperparameter_tuning_random_state=hyperparameter_tuning_random_state,
                id_columns_for_prediction=id_columns_for_prediction,
                split_stratify=split_stratify,
            )

            return self.result

    def save(
        self,
        output_dir_path: str,
        project_name: str = "",
        save_user_scripts: bool = True,
        save_dev_scripts: bool = True,
        save_datasets: bool = False,
        save_run_info: bool = True,
        save_running_arguments: bool = False,
        add_explain: bool = True,
        cancel: Optional[CancellationToken] = None,
    ):
        self.result.save(
            output_dir_path=output_dir_path,
            project_name=project_name,
            save_user_scripts=save_user_scripts,
            save_dev_scripts=save_dev_scripts,
            save_datasets=save_datasets,
            save_run_info=save_run_info,
            save_running_arguments=save_running_arguments,
        )
        if add_explain:
            self._generator.save(
                result=self.result,
                output_dir_path=output_dir_path,
                project_name=project_name,
                cancel=cancel,
            )
