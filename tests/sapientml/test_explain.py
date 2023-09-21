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
import os
import pickle
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sapientml.main import SapientML
from sapientml.params import Task
from sapientml_core import SapientMLConfig

fxdir = Path("tests/fixtures").absolute()


@pytest.fixture(scope="function")
def test_data():
    return pd.read_csv(fxdir / "datasets" / "testdata_df.csv")


@pytest.fixture(scope="function")
def setup_request_parameters():
    def _request_parameters():
        task = Task(
            target_columns=["target_number"],
            task_type="regression",
            split_method="random",
            split_seed=17,
            split_train_size=0.75,
            split_column_name=None,
            time_split_num=5,
            time_split_index=4,
            adaptation_metric="r2",
            split_stratification=False,
        )
        config = SapientMLConfig()
        with open(fxdir / "params" / "dataset.pkl", mode="rb") as f:
            dataset = pickle.load(f)
        return task, config, dataset

    return _request_parameters


@pytest.fixture(scope="function")
def make_tempdir(dir=fxdir / "outputs"):
    with tempfile.TemporaryDirectory(dir=dir) as temp_dir_path_str:
        temp_dir = Path(temp_dir_path_str).absolute()
        temp_dir.mkdir(exist_ok=True)
        yield temp_dir


def test_explain(test_data, make_tempdir, caplog):
    output_script_name = "final_script.ipynb"
    logging.disable(logging.NOTSET)
    cls_ = SapientML(
        n_models=2,
        target_columns=["target_category_binary_num"],
        task_type="classification",
        id_columns_for_prediction=["explanatory_Id"],
        split_stratification=True,
        add_explanation=True,
    )
    cls_.fit(
        test_data.drop(["target_number_large_scale", "target_number_large_scale_neg"], axis=1),
        ignore_columns=["target_number_copy"],
        output_dir=make_tempdir.as_posix(),
    )
    assert os.path.exists(make_tempdir / output_script_name)
    assert "Error" not in caplog.text
    with open(make_tempdir / output_script_name, "rt", encoding="UTF-8") as f:
        text = f.read()
    assert "General Structure" in text
    assert "Exploratory Data Analysis (EDA)" in text
    assert "Skewness" in text
    assert "Finding Intresting Datapoints" in text
    assert "Visualization for data distribution of columns" in text
    assert "Visualization for feature heatmap" in text
    assert "target_number_copy" not in text.split("# LOAD DATA")[1].split("# DROP IGNORED COLUMNS")[0]
    caplog.clear()
    logging.disable(logging.FATAL)


def test_explain_specify_train_test(test_data, make_tempdir, caplog):
    output_script_name = "final_script.ipynb"
    test_data = test_data.drop(["target_number_large_scale", "target_number_large_scale_neg"], axis=1)
    train_df = test_data[:200]
    test_df = test_data[200:]
    logging.disable(logging.NOTSET)
    cls_ = SapientML(
        n_models=2,
        target_columns=["target_category_binary_num"],
        id_columns_for_prediction=["explanatory_Id"],
        task_type="classification",
        split_stratification=True,
        add_explanation=True,
    )
    cls_.fit(
        training_data=train_df,
        test_data=test_df,
        ignore_columns=["target_number_copy"],
        output_dir=make_tempdir.as_posix(),
    )
    assert os.path.exists(make_tempdir / output_script_name)
    assert "Error" not in caplog.text
    with open(make_tempdir / output_script_name, "rt", encoding="UTF-8") as f:
        text = f.read()
    assert "General Structure" in text
    assert "Exploratory Data Analysis (EDA)" in text
    assert "Skewness" in text
    assert "Finding Intresting Datapoints" in text
    assert "Visualization for data distribution of columns" in text
    assert "Visualization for feature heatmap" in text
    assert "target_number_copy" not in text.split("# LOAD DATA")[1].split("# DROP IGNORED COLUMNS")[0]
    caplog.clear()
    logging.disable(logging.FATAL)
