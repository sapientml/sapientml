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
import pickle
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sapientml.main import SapientML
from sapientml.util.logging import setup_logger

fxdir = Path("tests/fixtures").absolute()


@pytest.fixture(scope="function")
def testdata_df_light():
    return pd.read_csv(fxdir / "datasets" / "testdata_df_light.csv")


@pytest.fixture(scope="function")
def testdata_df():
    return pd.read_csv(fxdir / "datasets" / "testdata_df.csv")


@pytest.fixture(scope="function")
def create_missing_dataframe():
    def _create_missing_dataframe(dataframe, missing_col_dict):
        dataframe = dataframe.reset_index(drop=True)
        for col, missing_ratio in missing_col_dict.items():
            process_index = dataframe.sample(frac=missing_ratio, random_state=0).index
            if col in dataframe.columns:
                dataframe.loc[process_index, col] = np.nan
        return dataframe

    return _create_missing_dataframe


def test_sapientml_works_with_two_targets(testdata_df_light):
    cls_ = SapientML(
        ["target_number", "target_number_large_scale"],
        task_type="regression",
        initial_timeout=60,
    )
    cls_.fit(
        testdata_df_light,
    )


def test_sapientml_works_with_suggestion(testdata_df_light):
    cls_ = SapientML(
        ["target_number"],
        initial_timeout=60,
    )
    cls_.fit(
        testdata_df_light,
        codegen_only=True,
    )


def test_sapientml_works_with_generated_model(testdata_df_light):
    cls_ = SapientML(
        ["target_number"],
        task_type="regression",
        initial_timeout=60,
    )
    cls_.fit(
        testdata_df_light,
        codegen_only=True,
    )
    model = cls_.model
    X = testdata_df_light.drop(["target_number"], axis=1)
    y = testdata_df_light["target_number"]
    model.fit(X, y)
    model.predict(X)


def test_sapientml_works_with_pickled_model_bytes_like_object(testdata_df_light):
    cls_ = SapientML(
        ["target_number"],
        task_type="regression",
        initial_timeout=60,
    )
    cls_.fit(
        testdata_df_light,
    )
    pkl = pickle.dumps(cls_.model)
    cls_ = SapientML.from_pretrained(pkl)
    cls_.predict(
        testdata_df_light,
    )


def test_sapientml_works_with_pickled_model_filename(testdata_df_light):
    cls_ = SapientML(
        ["target_number"],
        task_type="regression",
        initial_timeout=60,
    )
    cls_.fit(
        testdata_df_light,
    )
    with open("model.pkl", "wb") as f:
        pickle.dump(cls_.model, f)
    cls_ = SapientML.from_pretrained("model.pkl")
    cls_.predict(
        testdata_df_light,
    )


def test_sapientml_works_with_pickled_model(testdata_df_light):
    cls_ = SapientML(
        ["target_number"],
        task_type="regression",
        initial_timeout=60,
    )
    cls_.fit(
        testdata_df_light,
    )
    with open("model.pkl", "wb") as f:
        pickle.dump(cls_.model, f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    cls_ = SapientML.from_pretrained(model)
    cls_.predict(
        testdata_df_light,
    )


def test_sapientml_works_with_probability_prediction_for_multiclass_with_id(testdata_df_light):
    cls_ = SapientML(
        ["target_category_multi_nonnum"],
        task_type="classification",
        id_columns_for_prediction=["id"],
        initial_timeout=60,
    )
    test_df = testdata_df_light.copy()
    test_df["id"] = np.arange(test_df.shape[0])
    cls_.fit(
        test_df,
    )
    cls_.predict(
        test_df,
    )


def test_misc_sapientml_with_hpo_works(testdata_df_light, caplog):
    testdata_df_light = testdata_df_light[["target_number", "explanatory_multi_category_nonnum"]]
    logging.disable(logging.NOTSET)
    cls_ = SapientML(
        ["target_number"],
        task_type="regression",
        n_models=14,
        hyperparameter_tuning=True,
        hyperparameter_tuning_n_trials=1,
        hyperparameter_tuning_timeout=120,
    )
    cls_.fit(
        testdata_df_light,
    )
    assert "Error" not in caplog.text
    caplog.clear()
    logging.disable(logging.FATAL)


def test_sapientml_raises_error_if_undefined_tasktype_is_specified(testdata_df_light):
    with pytest.raises(ValueError):
        SapientML(
            ["target_number"],
            task_type="undefined_task_type",  # type: ignore
        )


def test_sapientml_raises_error_if_undefined_metric_is_specified(testdata_df_light):
    with pytest.raises(ValueError):
        SapientML(
            ["target_number"],
            task_type="regression",
            adaptation_metric="undefined_metric",
        )


def test_sapientml_works_with_group_split(testdata_df_light):
    cls_ = SapientML(
        ["target_number"],
        split_method="group",
        split_column_name="explanatory_groupId",
        task_type="regression",
    )
    cls_.fit(
        testdata_df_light,
    )


def test_sapientml_works_with_time_split(testdata_df_light):
    # Check if the class works with no exceptions
    # FIXME This testcase has multiple targets
    cls_ = SapientML(
        ["target_number"],
        task_type="regression",
        split_method="time",
        split_column_name="explanatory_datetime",
        time_split_num=4,
        time_split_index=0,
    )

    # The configuration is not correct, so the script has an error
    cls_.fit(
        testdata_df_light,
    )


def test_sapientml_raises_error_if_all_candidates_failed_to_run(testdata_df_light):
    cls_ = SapientML(
        ["target_number"],
        task_type="regression",
        split_method="time",
        split_column_name="explanatory_datetime",
        time_split_num=4,
        time_split_index=0,
    )
    with mock.patch("asyncio.create_subprocess_exec") as process:
        attrs = {
            "return_value.stdout.readline.return_value": (b""),
            "return_value.stderr.readline.return_value": (b""),
            "return_value.stdout.at_eof.return_value": True,
            "return_value.stderr.at_eof.return_value": True,
            "return_value.returncode": 1,
        }
        process.configure_mock(**attrs)
        with pytest.raises(RuntimeError):
            cls_.fit(
                testdata_df_light,
            )


def test_misc_sapientml_set_logger_handler_only_once():
    logger = setup_logger()
    assert len(logger.handlers) == 1
    _ = SapientML([""])
    logger = setup_logger()
    assert len(logger.handlers) == 1
    _ = SapientML([""])
    assert len(logger.handlers) == 1


def test_sapientml_raises_error_if_target_has_inf(testdata_df_light):
    import numpy as np

    testdata_df_light.loc[[1, 4, 7, 9, 11, 16, 19, 20], "target_number"] = np.inf
    with pytest.raises(Exception):
        cls_ = SapientML(
            ["target_number"],
            task_type="regression",
        )

        cls_.fit(
            testdata_df_light,
        )


def test_misc_sapientml_set_default_adaptation_metric_in_regression(testdata_df_light):
    cls_ = SapientML(
        ["target_number"],
        task_type="regression",
    )
    cls_.fit(
        testdata_df_light,
    )
    assert cls_.task.adaptation_metric == "r2"


def test_misc_sapientml_set_default_adaptation_metric_classification(testdata_df_light):
    cls_ = SapientML(
        ["target_category_binary_num"],
        task_type="classification",
    )
    cls_.fit(
        testdata_df_light,
    )
    assert cls_.task.adaptation_metric == "f1"


def test_sapientml_works_with_timeCol_incorrect_value(testdata_df_light):
    testdata_df_light_str = testdata_df_light.copy()
    testdata_df_light_str["explanatory_datetime"] = testdata_df_light_str["explanatory_datetime"].astype(str)
    testdata_df_light_str.loc[[2, 4, 6, 8, 10], "explanatory_datetime"] = " "
    cls1_ = SapientML(
        ["target_number"],
        task_type="regression",
    )

    cls1_.fit(
        testdata_df_light_str,
    )

    testdata_df_light_date = testdata_df_light.copy()
    testdata_df_light_date["explanatory_datetime"] = pd.to_datetime(testdata_df_light_date["explanatory_datetime"])
    assert pd.api.types.is_datetime64_any_dtype(testdata_df_light_date["explanatory_datetime"])

    testdata_df_light_date.loc[[2, 4, 6, 8, 10], "explanatory_datetime"] = " "
    cls2_ = SapientML(
        ["target_number"],
        task_type="regression",
    )

    cls2_.fit(
        testdata_df_light_date,
    )

    assert cls1_.generator._best_pipeline_score == cls2_.generator._best_pipeline_score


def test_sapientml_works_with_ignored_mixed_type_column(testdata_df_light):
    cls_ = SapientML(
        ["target_category_binary_num"],
        ignore_columns=["explanatory_mixed_type"],
        task_type="classification",
    )
    cls_.fit(
        testdata_df_light,
    )


@pytest.mark.parametrize(
    ("use_pos_list", "use_word_stemming"),
    [
        (["名詞", "動詞", "助動詞", "形容詞", "副詞"], True),
        (None, False),
    ],
)
def test_sapientml_works_with_Japanese_text_column(testdata_df, use_pos_list, use_word_stemming):
    testdata_df.loc[1, "explanatory_text_japanese"] = 1

    cls_ = SapientML(
        ["target_number"],
        task_type="regression",
        use_pos_list=use_pos_list,
        use_word_stemming=use_word_stemming,
    )
    cls_.fit(
        testdata_df,
    )
    assert "# HANDLE JAPANESE TEXT" in cls_.generator._best_pipeline.test


@pytest.mark.parametrize(
    ("use_word_list", "expected_string"),
    [
        (["シェフ", "警察官"], "TfidfVectorizer(max_features=3000, vocabulary=use_word_list)"),
        (["シェフ", "犬"], "TfidfVectorizer(max_features=3000, vocabulary=use_word_list)"),
        (["シェフ", "シェフ"], "TfidfVectorizer(max_features=3000, vocabulary=use_word_list)"),
        (["猫", "犬"], "TfidfVectorizer(max_features=3000, vocabulary=use_word_list)"),
        (None, "TfidfVectorizer(max_features=3000)"),
        (
            {"explanatory_text_japanese": ["シェフ", "犬"], "not_exist_col": ["警察官"]},
            "TfidfVectorizer(max_features=3000, vocabulary=use_word_list.get(_col)",
        ),
        ({"not_in_col": ["猫", "犬"]}, "TfidfVectorizer(max_features=3000, vocabulary=use_word_list.get(_col)"),
    ],
)
def test_sapientml_works_with_specified_word_ja(testdata_df, use_word_list, expected_string):
    cls_ = SapientML(
        ["target_number"],
        task_type="regression",
        use_word_list=use_word_list,
    )
    cls_.fit(
        testdata_df,
    )
    assert (expected_string in cls_.generator._best_pipeline.test) and (
        expected_string in cls_.generator._best_pipeline.train
    )


@pytest.mark.parametrize(
    ("use_word_list", "expected_string"),
    [
        (["Chef", "police", "chef"], "TfidfVectorizer(max_features=3000, vocabulary=use_word_list)"),
        (None, "TfidfVectorizer(max_features=3000)"),
        (
            {"explanatory_text_english": ["police", "chef"], "not_exist_col": ["cat"]},
            "TfidfVectorizer(max_features=3000, vocabulary=use_word_list.get(_col)",
        ),
    ],
)
def test_sapientml_works_with_specified_word_en(testdata_df, use_word_list, expected_string):
    cls_ = SapientML(
        ["target_number"],
        task_type="regression",
        use_word_list=use_word_list,
    )
    cls_.fit(
        testdata_df,
    )
    assert (expected_string in cls_.generator._best_pipeline.test) and (
        expected_string in cls_.generator._best_pipeline.train
    )


@pytest.mark.parametrize("exp_col", ["explanatory_json", "explanatory_list"])
def test_sapientml_works_with_list_values(testdata_df, exp_col, caplog):
    logging.disable(logging.NOTSET)
    testdata_df = testdata_df[["target_category_multi_num", exp_col]]
    cls_ = SapientML(
        ["target_category_multi_num"],
        task_type="classification",
    )
    cls_.fit(
        testdata_df,
    )
    returncode = 1
    for i in range(len(cls_.generator.execution_results)):
        if cls_.generator.execution_results[i][1].returncode == 0:
            returncode = 0
            break
    assert returncode == 0
    caplog.clear()

    train_df = testdata_df[:200].reset_index(drop=True)
    valid_df = testdata_df[200:250].reset_index(drop=True)
    test_df = testdata_df[250:].reset_index(drop=True)
    cls_ = SapientML(
        ["target_category_multi_num"],
        task_type="classification",
    )
    cls_.fit(
        train_df,
        validation_data=valid_df,
        test_data=test_df,
    )
    returncode = 1
    for i in range(len(cls_.generator.execution_results)):
        if cls_.generator.execution_results[i][1].returncode == 0:
            returncode = 0
            break
    assert returncode == 0
    caplog.clear()
    logging.disable(logging.FATAL)


def test_sapientml_raises_error_if_target_columns_are_missing_for_regression():
    with pytest.raises(ValueError):
        SapientML(
            [],
            task_type="regression",
        )


def test_sapientml_raises_error_if_target_columns_are_missing_for_classification():
    with pytest.raises(ValueError):
        SapientML(
            [],
            task_type="classification",
        )


def test_sapientml_raises_error_if_target_columns_are_different_for_regression(testdata_df_light):
    cls_ = SapientML(
        ["s1"],
        task_type="regression",
    )
    with pytest.raises(Exception):
        cls_.fit(
            testdata_df_light,
        )


def test_sapientml_raises_error_if_target_columns_are_different_for_classification(testdata_df_light):
    cls_ = SapientML(
        ["s1"],
        task_type="classification",
    )
    with pytest.raises(Exception):
        cls_.fit(
            testdata_df_light,
        )


def test_sapientml_raises_error_if_columns_name_length_is_above_limit(testdata_df_light):
    testdata_df_light.rename(columns={"explanatory_groupId": "ID" * 1000}, inplace=True)
    with pytest.raises(Exception):
        cls_ = SapientML(
            ["target_number"],
            task_type="regression",
        )
        cls_.fit(
            testdata_df_light,
        )


def test_sapientml_raises_error_if_target_columns_are_multiple_with_stratification_true(testdata_df_light):
    cls_ = SapientML(
        ["target_category_binary_nonnum", "target_category_multi_nonnum"],
        task_type="classification",
        split_stratification=True,
    )
    with pytest.raises(Exception):
        cls_.fit(
            testdata_df_light,
        )


def test_sapientml_raises_error_if_target_has_nan(testdata_df_light):
    with pytest.raises(Exception):
        cls_ = SapientML(
            ["target_number_has_nan"],
            task_type="regression",
        )
        cls_.fit(
            testdata_df_light,
        )


def test_sapientml_raises_error_if_number_models_are_zero():
    with pytest.raises(Exception):
        SapientML(["target_number"], task_type="regression", n_models=0)


def test_misc_sapientml_with_hpo_works_for_classification_task(testdata_df_light, caplog):
    logging.disable(logging.NOTSET)
    testdata_df_light = testdata_df_light[
        ["target_category_binary_num", "explanatory_number", "explanatory_multi_category_nonnum"]
    ]
    cls_ = SapientML(
        ["target_category_binary_num"],
        task_type="classification",
        n_models=16,
        hyperparameter_tuning=True,
        hyperparameter_tuning_n_trials=1,
        hyperparameter_tuning_timeout=120,
    )
    cls_.fit(
        testdata_df_light,
    )

    assert "Error" not in caplog.text
    caplog.clear()
    logging.disable(logging.FATAL)


def test_sapientml_works_with_classification_split_stratification(testdata_df_light):
    cls_ = SapientML(
        ["target_category_binary_num"],
        task_type="classification",
        split_stratification=True,
        adaptation_metric="LogLoss",
    )
    cls_.fit(
        testdata_df_light,
    )

    assert "stratify" in cls_.generator._best_pipeline.test


def test_sapientml_works_with_regression_split_stratification(testdata_df_light):
    cls_ = SapientML(
        ["target_number"],
        task_type="regression",
        split_stratification=True,
    )
    cls_.fit(
        testdata_df_light,
    )

    assert "stratify" not in cls_.generator._best_pipeline.test


def test_sapientml_works_with_symbol_column(testdata_df_light):
    col_has_symbol = {"target_number_large_scale": "[target_number]{}:<\\+"}
    testdata_df_light = testdata_df_light.rename(columns=col_has_symbol)

    cls_ = SapientML(
        target_columns=["[target_number]{}:<\\+", "target_number"],
        add_explanation=True,
    )
    cls_.fit(
        testdata_df_light,
    )


def test_sapientml_works_with_change_none_to_nan():
    df = pd.DataFrame(
        {
            "A": list(range(1, 11)),
            "B": [None if i % 2 == 0 else "B" for i in range(1, 11)],
            # It also fails when the value np.nan is included.
            # "B": [np.nan if i % 2 == 0 else "B" for i in range(1, 11)],
            "y": ["y" if i % 2 == 0 else "n" for i in range(1, 11)],
        }
    )

    sml = SapientML(
        ["y"],
        add_explanation=True,
    )

    sml.fit(df)
