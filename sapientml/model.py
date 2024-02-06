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
import shutil
import tempfile
import time
from os import PathLike
from pathlib import Path
from typing import Literal, Optional, Union
from uuid import uuid4 as uuid

import lancedb
import pandas as pd
from lancedb.pydantic import LanceModel, Vector

from .executor import run
from .params import save_file
from .util.logging import setup_logger

logger = setup_logger()


def _readfile(files, filepath: Path, input_dir: Path):
    _, ext = os.path.splitext(filepath)
    if ext == ".pkl":
        with open(filepath, "rb") as f:
            files.append((str(filepath.relative_to(input_dir)), " ".join(map(str, f.read()))))
    else:
        with open(filepath, "r") as f:
            files.append((str(filepath.relative_to(input_dir)), f.read()))


class GeneratedModel(LanceModel):
    id: str
    prev_id: str | None = None
    vector: Vector(1)  # 1-d vector containing timestamp
    files: list[tuple[str, str]]
    save_datasets_format: str
    timeout: int
    csv_encoding: str
    csv_delimiter: str
    params: list[tuple[str, str, str]]
    score: float
    metric: str

    @staticmethod
    def create(
        input_dir: PathLike,
        save_datasets_format: Literal["csv", "pickle"],
        timeout: int,
        csv_encoding: str,
        csv_delimiter: str,
        params: dict,
    ):
        """
        The factory method of GeneratedModel.
        Instantiating this class by yourself is not intended.

        Parameters
        ----------
        input_dir: PathLike
            Directory path containing training/prediction scripts and trained models.
        save_datasets_format: 'csv' or 'pickle'
            Data format when the input dataframes are written to files.
            Ignored when all inputs are specified as file path.
        timeout: int
            Timeout for the execution of training and prediction.
        csv_encoding: str
            Encoding method when csv files are involved.
            Ignored when only pickle files are involved.
        csv_delimiter: str
            Delimiter to read csv files.
        """
        input_dir = Path(input_dir)
        files = list()
        _readfile(files, input_dir / "final_script.py", input_dir)
        _readfile(files, input_dir / "final_train.py", input_dir)
        _readfile(files, input_dir / "final_predict.py", input_dir)

        for filepath in input_dir.glob("lib/*.py"):
            _readfile(files, filepath, input_dir)

        for filepath in input_dir.glob("**/*.pkl"):
            if save_datasets_format == "pickle" and "training.pkl" == filepath.name:
                continue
            _readfile(files, filepath, input_dir)

        with open(input_dir / "final_script.out.json", "r") as f:
            data = json.load(f)
            score = float(data["score"])
            metric = data["metric"]

        id = str(uuid())

        return GeneratedModel(
            id=id,
            vector=[time.time()],
            files=files,
            save_datasets_format=save_datasets_format,
            timeout=timeout,
            csv_encoding=csv_encoding,
            csv_delimiter=csv_delimiter,
            params=[(k, str(type(v)), str(v)) for k, v in params.items()],
            score=score,
            metric=metric,
        )

    def save(self):
        """
        Save itself to database
        """
        db = lancedb.connect(".lancedb")
        table = db.create_table("models", schema=GeneratedModel, exist_ok=True)
        table.add([self])
        return self.id

    @staticmethod
    def load(id):
        """
        Load a GeneratedModel with id from database
        """
        db = lancedb.connect(".lancedb")
        table = db.open_table("models")
        model = table.search().where(f"id='{id}'").limit(1).to_pydantic(GeneratedModel)[0]
        model.prev_id = id
        model.id = str(uuid())
        return model

    def export(self, output_dir: Union[str, PathLike]):
        """
        Export generated code to `output_dir` folder

        Parameters
        ----------
        output_dir: Path-like object
            Training dataframe.

        Returns
        -------
        self: GeneratedModel
            GeneratedModel object itself
        """
        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        (output_dir / "lib").mkdir(exist_ok=True)
        for filename, content in self.files:
            _, ext = os.path.splitext(filename)
            if ext == ".pkl":
                with open(output_dir / filename, "wb") as f:
                    f.write(b"".join([int(i).to_bytes(1, "little") for i in content.split(" ")]))
            else:
                with open(output_dir / filename, "w") as f:
                    f.write(content)

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series]] = None):
        """
        Generate ML scripts for input data.

        Parameters
        ----------
        X: pandas.DataFrame
            Training dataframe. Contains target values if `y` is `None`.
        y: pandas.DataFrame or pandas.Series
            The target values.

        Returns
        -------
        self: GeneratedModel
            GeneratedModel object itself
        """
        if y is not None:
            X = pd.concat([X, y], axis=1)

        with tempfile.TemporaryDirectory() as temp_dir_path_str:
            temp_dir = Path(temp_dir_path_str).absolute()
            temp_dir.mkdir(exist_ok=True)
            self.export(temp_dir)
            filename = "training." + ("pkl" if self.save_datasets_format == "pickle" else "csv")
            save_file(X, str(temp_dir / filename), self.csv_encoding, self.csv_delimiter)
            logger.info("Building model by generated pipeline...")
            result = run(str(temp_dir / "final_train.py"), self.timeout)
            if result.returncode != 0:
                raise RuntimeError(f"Training was failed due to the following Error: {result.error}")
            for filepath in temp_dir.glob("**/*.pkl"):
                if self.save_datasets_format == "pickle" and "training.pkl" == filepath.name:
                    continue
                _readfile(self.files, filepath, temp_dir)
        return self

    def predict(self, X: pd.DataFrame):
        """Predicts the output of the test_data and store in the prediction_result.csv.

        Parameters
        ---------
        X: pd.DataFrame
            Dataframe used for predicting the result.

        Returns
        -------
        result_df : pd.DataFrame
            It returns the prediction_result.csv result in dataframe format.
        """
        with tempfile.TemporaryDirectory() as temp_dir_path_str:
            temp_dir = Path(temp_dir_path_str).absolute()
            temp_dir.mkdir(exist_ok=True)
            self.export(temp_dir)
            filename = "test." + ("pkl" if self.save_datasets_format == "pickle" else "csv")
            save_file(X, str(temp_dir / filename), self.csv_encoding, self.csv_delimiter)
            result = run(str(temp_dir / "final_predict.py"), self.timeout)
            if result.returncode != 0:
                raise RuntimeError(f"Prediction was failed due to the following Error: {result.error}")
            id_columns_for_prediction = None
            for k, _, v in self.params:
                if k == "id_columns_for_prediction":
                    id_columns_for_prediction = eval(v)
                    break
            if id_columns_for_prediction is None:
                id_columns_for_prediction = [0]
            result_df = pd.read_csv(temp_dir / "prediction_result.csv", index_col=id_columns_for_prediction)
            return result_df
