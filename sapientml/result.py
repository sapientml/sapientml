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

import json
import os
from pathlib import Path
from shutil import copyfile
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel

from .params import CancellationToken, Code, PipelineResult
from .util.json_util import JSONEncoder
from .util.logging import setup_logger

logger = setup_logger()


class SapientMLGeneratorResult(BaseModel):
    final_script: Optional[tuple[Optional[Code], PipelineResult]]
    candidate_scripts: list[tuple[Code, PipelineResult]]
    training_data: pd.DataFrame
    validation_data: Optional[pd.DataFrame]
    test_data: Optional[pd.DataFrame]
    training_data_path: Optional[str]
    validation_data_path: Optional[str]
    test_data_path: Optional[str]
    csv_encoding: Literal["UTF-8", "SJIS"]
    tmpdir_path: Path
    target_columns: list[str]
    task_type: Literal["classification", "regression"]
    ignore_columns: list[str]
    split_method: str
    split_seed: int
    split_train_size: float
    split_column_name: Optional[str]
    time_split_num: int
    time_split_index: int
    n_models: int
    seed_for_model: int
    adaptation_metric: Optional[str]
    csv_delimiter: str
    initial_timeout: int
    timeout_for_test: int
    use_hyperparameters: bool
    predict_option: Literal["default", "probability"]
    hyperparameter_tuning: bool
    hyperparameter_tuning_n_trials: int
    hyperparameter_tuning_timeout: int
    hyperparameter_tuning_random_state: int
    id_columns_for_prediction: list[str]
    split_stratify: bool

    class Config:
        arbitrary_types_allowed = True

    def save(
        self,
        output_dir_path: str,
        project_name: str = "",
        save_user_scripts: bool = True,
        save_dev_scripts: bool = True,
        save_datasets: bool = False,
        save_run_info: bool = True,
        save_running_arguments: bool = False,
        cancel: Optional[CancellationToken] = None,
    ):
        """
        Save generated scripts and datasets.

        Parameters
        ----------
        output_dir_path: str
            The output directory path.
        project_name: str
            Prefix to file names.
        save_user_scripts: bool
            Output the final script (*final_default.py).
        save_dev_scripts: bool
            Output the candidates scripts such as 1_default.py.
        save_datasets: bool
            Output the datasets to files if file paths are input to generate_code().
        save_run_info: bool
            Output the result file of evaluation (*run_info.json).
        save_running_arguments: bool
            Output the settings (input.json).

        Returns
        -------
        None
        """

        def add_prefix(filename, prefix):
            if not prefix:
                return filename
            return f"{prefix}_{filename}"

        path = Path(output_dir_path)
        path.mkdir(parents=True, exist_ok=True)

        if save_dev_scripts and self.candidate_scripts:
            # copy libs
            lib_path = path / "lib"
            lib_path.mkdir(exist_ok=True)
            copyfile(
                Path(os.path.dirname(__file__)) / "../static/lib/sample_dataset.py", lib_path / "sample_dataset.py"
            )

            for index, (script, detail) in enumerate(self.candidate_scripts, start=1):
                # script.dataset.training_data_path is '{user specified dir}/{name}.csv' or '{tmpdir}/training.pkl'
                # If latter one, we have to modify the {tmpdir} to output_dir.
                script_body = script.validation.replace(self.tmpdir_path.as_posix(), ".")

                with open(path / f"{index}_script.py", "w", encoding="utf-8") as f:
                    f.write(script_body)

        if save_datasets:
            # script.dataset.training_data_path is '{user specified dir}/{name}.csv' or '{tmpdir}/training.csv' or '{tmpdir}/training.pkl'
            # We want to save dataset only when the last two ones.
            training_data_path = self.training_data_path
            if Path(training_data_path).parent == self.tmpdir_path:
                if Path(training_data_path).suffix == ".pkl":
                    self.training_data.to_pickle(path / Path(training_data_path).name)
                else:
                    self.training_data.to_csv(
                        path / Path(training_data_path).name, encoding=self.csv_encoding, index=False
                    )
            validation_data_path = self.validation_data_path
            if validation_data_path and Path(validation_data_path).parent == self.tmpdir_path:
                if Path(validation_data_path).suffix == ".pkl":
                    self.validation_data.to_pickle(path / Path(validation_data_path).name)
                else:
                    self.validation_data.to_csv(
                        path / Path(validation_data_path).name, encoding=self.csv_encoding, index=False
                    )
            test_data_path = self.test_data_path
            if test_data_path and Path(test_data_path).parent == self.tmpdir_path:
                if Path(test_data_path).suffix == ".pkl":
                    self.test_data.to_pickle(path / Path(test_data_path).name)
                else:
                    self.test_data.to_csv(path / Path(test_data_path).name, encoding=self.csv_encoding, index=False)

        if save_run_info and self.candidate_scripts:
            if self.final_script:
                with open(
                    path / (add_prefix("final_script", project_name) + ".out.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(self.final_script[1].__dict__, f, cls=JSONEncoder, indent=4)
            else:
                raise RuntimeError("All candidate scripts failed. Final script is not saved.")

            debug_info = {}

            for i, candidate in enumerate(self.candidate_scripts, start=1):
                info = {"content": candidate[0].dict(), "run_info": candidate[1].__dict__}
                debug_info[i] = info

            with open(path / add_prefix("run_info.json", project_name), "w", encoding="utf-8") as f:
                json.dump(debug_info, f, cls=JSONEncoder, indent=4)

        if not self.final_script:
            logger.warning("All candidate scripts failed. Final script is not saved.")
            raise RuntimeError("All candidate scripts failed. Final script is not saved.")

        if save_user_scripts:
            # script.task.train_dataset_path is '{user specified dir}/{name}.csv' or '{tmpdir}/training.pkl'
            # If latter one, we have to modify the {tmpdir} to output_dir.
            script_body = self.final_script[0].test.replace(self.tmpdir_path.as_posix(), ".")
            with open(path / add_prefix("final_script.py", project_name), "w", encoding="utf-8") as f:
                f.write(script_body)

            script_body = self.final_script[0].train.replace(self.tmpdir_path.as_posix(), ".")
            with open(path / add_prefix("final_train.py", project_name), "w", encoding="utf-8") as f:
                f.write(script_body)

            script_body = self.final_script[0].predict.replace(self.tmpdir_path.as_posix(), ".")
            with open(path / add_prefix("final_predict.py", project_name), "w", encoding="utf-8") as f:
                f.write(script_body)

        if save_running_arguments:
            with open(path / "input.json", "w", encoding="utf-8") as f:
                kwargs = {
                    "target_columns": self.target_columns,
                    "task_type": self.task_type,
                    "ignore_columns": self.ignore_columns,
                    "split_method": self.split_method,
                    "split_seed": self.split_seed,
                    "split_train_size": self.split_train_size,
                    "split_column_name": self.split_column_name,
                    "time_split_num": self.time_split_num,
                    "time_split_index": self.time_split_index,
                    "n_models": self.n_models,
                    "seed_for_model": self.seed_for_model,
                    "adaptation_metric": self.adaptation_metric,
                    "csv_delimiter": self.csv_delimiter,
                    "initial_timeout": self.initial_timeout,
                    "timeout_for_test": self.timeout_for_test,
                    "use_hyperparameters": self.use_hyperparameters,
                    "predict_option": self.predict_option,
                    "hyperparameter_tuning": self.hyperparameter_tuning,
                    "hyperparameter_tuning_n_trials": self.hyperparameter_tuning_n_trials,
                    "hyperparameter_tuning_timeout": self.hyperparameter_tuning_timeout,
                    "hyperparameter_tuning_random_state": self.hyperparameter_tuning_random_state,
                    "id_columns_for_prediction": self.id_columns_for_prediction,
                    "split_stratify": self.split_stratify,
                }
                json.dump(kwargs, f, cls=JSONEncoder, indent=4)

        logger.info(f"The files saved in {path.absolute().as_posix()}")
