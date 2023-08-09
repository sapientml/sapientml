import logging
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sapientml.main import SapientML

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


def test_sapientml_works(testdata_df_light):
    cls_ = SapientML()
    ret = cls_.generate_code(
        training_data=testdata_df_light,
        target_columns=["target_number"],
        task_type="regression",
    )

    assert ret.final_script[0] and ret.final_script[1].score


def test_sapientml_works_with_two_target(testdata_df_light):
    cls_ = SapientML()
    ret = cls_.generate_code(
        training_data=testdata_df_light,
        target_columns=["target_number", "target_number_large_scale"],
        task_type="regression",
        initial_timeout=60,
    )

    assert ret.final_script[0] and ret.final_script[1].score


def test_sapientml_works_in_classification(testdata_df_light):
    cls_ = SapientML()
    ret = cls_.generate_code(
        training_data=testdata_df_light,
        target_columns=["target_category_binary_num"],
        task_type="classification",
    )

    assert ret.final_script[0] and ret.final_script[1].score


def test_sapientml_with_hpo_works(testdata_df_light, caplog):
    testdata_df_light = testdata_df_light[["target_number", "explanatory_multi_category_nonnum"]]
    logging.disable(logging.NOTSET)
    cls_ = SapientML()
    ret = cls_.generate_code(
        training_data=testdata_df_light,
        target_columns=["target_number"],
        task_type="regression",
        n_models=14,
        hyperparameter_tuning=True,
        hyperparameter_tuning_n_trials=1,
        hyperparameter_tuning_timeout=120,
    )
    assert ret.final_script[0] and "Error" not in caplog.text
    caplog.clear()
    logging.disable(logging.FATAL)


def test_raise_error_if_undefined_tasktype_is_specified(testdata_df_light):
    cls_ = SapientML()
    with pytest.raises(ValueError):
        cls_.generate_code(
            training_data=testdata_df_light,
            target_columns=["target_number"],
            task_type="undefined_task_type",  # type: ignore
        )


def test_raise_error_if_undefined_metric_is_specified(testdata_df_light):
    cls_ = SapientML()
    with pytest.raises(ValueError):
        cls_.generate_code(
            training_data=testdata_df_light,
            target_columns=["target_number"],
            task_type="regression",
            adaptation_metric="undefined_metric",
        )


def test_sapientml_works_with_group_split(testdata_df_light):
    cls_ = SapientML()
    ret = cls_.generate_code(
        training_data=testdata_df_light,
        target_columns=["target_number"],
        split_method="group",
        split_column_name="explanatory_groupId",
        task_type="regression",
    )
    assert ret.final_script[0] and ret.final_script[1].score


def test_sapientml_works_with_time_split(testdata_df_light):
    # Check if the class works with no exceptions
    # FIXME This testcase has multiple targets
    cls_ = SapientML()

    # The configuration is not correct, so the script has an error
    cls_.generate_code(
        training_data=testdata_df_light,
        target_columns=["target_number"],
        task_type="regression",
        split_method="time",
        split_column_name="explanatory_datetime",
        time_split_num=4,
        time_split_index=0,
    )


def test_sapientml_generate_code_returns_candidates_even_if_all_the_candidates_failed_to_run(testdata_df_light):
    cls_ = SapientML()
    with mock.patch("subprocess.Popen") as process:
        attrs = {"return_value.communicate.return_value": (b"", b"Error!"), "return_value.returncode": 1}
        process.configure_mock(**attrs)

        ret = cls_.generate_code(
            training_data=testdata_df_light,
            target_columns=["target_number"],
            task_type="regression",
            split_method="time",
            split_column_name="explanatory_datetime",
            time_split_num=4,
            time_split_index=0,
        )

    assert ret.candidate_scripts and ret.final_script[1].score is None


def test_sapientml_generate_code_returns_top_script_if_any_one_script_ran_successfully(testdata_df_light):
    cls_ = SapientML()
    with mock.patch("subprocess.Popen") as process:
        side_effects = [
            mock.Mock(communicate=mock.Mock(return_value=x[0]), returncode=x[1])
            for x in [
                [
                    (
                        b"RESULT: R2: 0.87654",
                        b"",
                    ),
                    0,
                ],
                [
                    (
                        b"",
                        b"",
                    ),
                    1,
                ],
            ]
        ]

        def _side_effect(*args, **kwargs):
            _value = min(process.call_count, len(side_effects))
            return side_effects[_value - 1]

        process.side_effect = _side_effect

        ret = cls_.generate_code(
            training_data=testdata_df_light,
            target_columns=["target_number"],
            task_type="regression",
            split_method="time",
            split_column_name="explanatory_datetime",
            time_split_num=4,
            time_split_index=0,
        )

    assert ret.final_script[0] and ret.final_script[1].score


def test_sapientml_set_logger_handler_only_once():
    logger = logging.getLogger("sapientml")
    assert len(logger.handlers) == 0
    _ = SapientML()
    logger = logging.getLogger("sapientml")
    assert len(logger.handlers) == 1
    _ = SapientML()
    assert len(logger.handlers) == 1


# @mock.patch("sapientml.common.macros.ROW_THRESHOLD_FOR_SAMPLING", 10)
# def test_sapientml_get_dev_training_for_adaptation_works(tmp_path, housing_dataframe):
#    cls_ = SapientML()
#    _, _, dataset, _ = _split_dataset(housing_dataframe)
#    sampled, _ = cls_._get_dev_training_for_adaptation(
#        dataset, "path/to/dataset", ["SalePrice"], "regression", tmp_path
#    )
#    assert len(sampled) == 10


def test_sapientml_raise_error_if_target_has_inf(testdata_df_light):
    import numpy as np

    testdata_df_light.loc[[1, 4, 7, 9, 11, 16, 19, 20], "target_number"] = np.inf
    with pytest.raises(Exception):
        cls_ = SapientML()

        cls_.generate_code(
            training_data=testdata_df_light,
            target_columns=["target_number"],
            task_type="regression",
        )


# def test_sapientml_ignore_columns_initialized_everytime(testdata_df_light):
#     class RunCalledException(Exception):
#         pass

#     class SummarizeDatasetCalledException(Exception):
#         pass

#     with mock.patch(
#         "sapientml.executor.run", side_effect=RunCalledException()
#     ) as run:
#         # add all null column to change ignore_columns
#         df = testdata_df_light.copy()
#         df["null_columns"] = None
#         with pytest.raises(RunCalledException):
#             cls_ = SapientML()
#             cls_.generate_code(
#                 training_data=df,
#                 target_columns=["target_number"],
#                 task_type="regression",
#             )
#         assert run.called

#     with mock.patch(
#         "sapientml.code_block_generator.prediction_based.params.summarize_dataset", side_effect=SummarizeDatasetCalledException()
#     ) as summarize_dataset:
#         with pytest.raises(SummarizeDatasetCalledException):
#             # cls_ = SapientML()
#             cls_.generate_code(
#                 training_data=testdata_df_light,
#                 target_columns=["target_number"],
#                 task_type="regression",
#             )

#         args, _ = summarize_dataset.call_args
#         _, task, _, _ = args
#         assert len(task.ignore_columns) == 0


def test_sapientml_adaptation_metric_is_None_regression(testdata_df_light):
    cls_ = SapientML()
    ret = cls_.generate_code(
        training_data=testdata_df_light,
        target_columns=["target_number"],
        task_type="regression",
    )
    assert ret.adaptation_metric == "r2"


def test_sapientml_adaptation_metric_is_None_classification(testdata_df_light):
    cls_ = SapientML()
    ret = cls_.generate_code(
        training_data=testdata_df_light,
        target_columns=["target_category_binary_num"],
        task_type="classification",
    )
    assert ret.adaptation_metric == "f1"


def test_sapientml_works_with_timeCol_incorrect_value(testdata_df_light):
    testdata_df_light_str = testdata_df_light.copy()
    testdata_df_light_str["explanatory_datetime"] = testdata_df_light_str["explanatory_datetime"].astype(str)
    testdata_df_light_str.loc[[2, 4, 6, 8, 10], "explanatory_datetime"] = " "
    cls_ = SapientML()

    ret1 = cls_.generate_code(
        training_data=testdata_df_light_str,
        target_columns=["target_number"],
        task_type="regression",
    )
    assert ret1.final_script and ret1.final_script[1].score

    testdata_df_light_date = testdata_df_light.copy()
    testdata_df_light_date["explanatory_datetime"] = pd.to_datetime(testdata_df_light_date["explanatory_datetime"])
    assert pd.api.types.is_datetime64_any_dtype(testdata_df_light_date["explanatory_datetime"])

    testdata_df_light_date.loc[[2, 4, 6, 8, 10], "explanatory_datetime"] = " "
    cls_ = SapientML()

    ret2 = cls_.generate_code(
        training_data=testdata_df_light_date,
        target_columns=["target_number"],
        task_type="regression",
    )
    assert ret2.final_script and ret2.final_script[1].score
    assert ret1.final_script[1].score == ret2.final_script[1].score


def test_sapientml_works_with_ignored_mixed_type_column(testdata_df_light):
    cls_ = SapientML()
    ret = cls_.generate_code(
        training_data=testdata_df_light,
        ignore_columns=["explanatory_mixed_type"],
        target_columns=["target_category_binary_num"],
        task_type="classification",
    )
    assert ret.final_script[0] and ret.final_script[1].score


def test_sapientml_works_with_Japanese_download_model(testdata_df):
    src_root = Path(__file__).parents[2]
    model_path = src_root / "code_block_generator" / "rule_based" / "lib" / "lid.176.bin"
    if model_path.exists():
        model_path.unlink()
    testdata_df.loc[1, "explanatory_text_japanese"] = 1

    cls_ = SapientML()
    ret1 = cls_.generate_code(
        training_data=testdata_df,
        target_columns=["target_number"],
        task_type="regression",
    )
    assert "# HANDLE JAPANESE TEXT" in ret1.final_script[0].code_for_test


@pytest.mark.parametrize(
    ("use_pos_list", "use_word_stemming"),
    [
        (["名詞", "動詞", "助動詞", "形容詞", "副詞"], True),
        (None, False),
    ],
)
def test_sapientml_works_with_Japanese_text_column(testdata_df, use_pos_list, use_word_stemming):
    testdata_df.loc[1, "explanatory_text_japanese"] = 1

    cls_ = SapientML()
    ret1 = cls_.generate_code(
        training_data=testdata_df,
        target_columns=["target_number"],
        task_type="regression",
        use_pos_list=use_pos_list,
        use_word_stemming=use_word_stemming,
    )
    assert "# HANDLE JAPANESE TEXT" in ret1.final_script[0].code_for_test


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
    cls_ = SapientML()
    ret = cls_.generate_code(
        training_data=testdata_df,
        target_columns=["target_number"],
        task_type="regression",
        use_word_list=use_word_list,
    )
    assert (expected_string in ret.final_script[0].code_for_test) and (
        expected_string in ret.final_script[0].code_for_train
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
    cls_ = SapientML()
    ret = cls_.generate_code(
        training_data=testdata_df,
        target_columns=["target_number"],
        task_type="regression",
        use_word_list=use_word_list,
    )
    assert (expected_string in ret.final_script[0].code_for_test) and (
        expected_string in ret.final_script[0].code_for_train
    )


@pytest.mark.parametrize("exp_col", ["explanatory_json", "explanatory_list"])
def test_sapientml_works_with_list_values(testdata_df, exp_col, caplog):
    logging.disable(logging.NOTSET)
    testdata_df = testdata_df[["target_category_multi_num", exp_col]]
    cls_ = SapientML()
    ret1 = cls_.generate_code(
        training_data=testdata_df,
        target_columns=["target_category_multi_num"],
        task_type="classification",
    )
    ret1.save(
        "./tests/fixtures/outputs",
        save_dev_scripts=True,
        save_user_scripts=True,
        save_datasets=True,
        save_running_arguments=True,
    )
    assert ret1.final_script and ret1.final_script[1].score and "Error" not in caplog.text
    caplog.clear()

    train_df = testdata_df[:200].reset_index(drop=True)
    valid_df = testdata_df[200:250].reset_index(drop=True)
    test_df = testdata_df[250:].reset_index(drop=True)
    cls_ = SapientML()
    ret2 = cls_.generate_code(
        training_data=train_df,
        validation_data=valid_df,
        test_data=test_df,
        target_columns=["target_category_multi_num"],
        task_type="classification",
    )
    ret2.save(
        "./tests/fixtures/outputs",
        save_dev_scripts=True,
        save_user_scripts=True,
        save_datasets=True,
        save_running_arguments=True,
    )
    assert ret2.final_script and ret1.final_script[1].score and "Error" not in caplog.text
    caplog.clear()
    logging.disable(logging.FATAL)


def test_raise_error_if_target_columns_are_missing_for_regression(testdata_df_light):
    cls_ = SapientML()
    with pytest.raises(ValueError):
        cls_.generate_code(
            training_data=testdata_df_light,
            target_columns=[],
            task_type="regression",
        )


def test_raise_error_if_target_columns_are_missing_for_classification(testdata_df_light):
    cls_ = SapientML()
    with pytest.raises(ValueError):
        cls_.generate_code(
            training_data=testdata_df_light,
            target_columns=[],
            task_type="classification",
        )


def test_raise_error_if_target_columns_are_different_for_regression(testdata_df_light):
    cls_ = SapientML()
    with pytest.raises(Exception):
        cls_.generate_code(
            training_data=testdata_df_light,
            target_columns=["s1"],
            task_type="regression",
        )


def test_raise_error_if_target_columns_are_different_for_classification(testdata_df_light):
    cls_ = SapientML()
    with pytest.raises(Exception):
        cls_.generate_code(
            training_data=testdata_df_light,
            target_columns=["s1"],
            task_type="classification",
        )


def test_raise_error_if_columns_name_length_is_above_limit(testdata_df_light):
    testdata_df_light.rename(columns={"explanatory_groupId": "ID" * 1000}, inplace=True)
    with pytest.raises(Exception):
        cls_ = SapientML()
        cls_.generate_code(
            training_data=testdata_df_light,
            target_columns=["target_number"],
            task_type="regression",
        )


def test_raise_error_if_target_columns_are_multiple_with_stratification_true(testdata_df_light):
    cls_ = SapientML()
    with pytest.raises(Exception):
        cls_.generate_code(
            training_data=testdata_df_light,
            target_columns=["target_category_binary_nonnum", "target_category_multi_nonnum"],
            task_type="classification",
            split_stratification=True,
        )


def test_sapientml_raise_error_if_target_has_nan(testdata_df_light):
    with pytest.raises(Exception):
        cls_ = SapientML()
        cls_.generate_code(
            training_data=testdata_df_light,
            target_columns=["target_number_has_nan"],
            task_type="regression",
        )


def test_sapientml_raise_error_if_number_models_are_zero(testdata_df_light):
    cls_ = SapientML()
    with pytest.raises(Exception):
        cls_.generate_code(
            training_data=testdata_df_light, target_columns=["target_number"], task_type="regression", n_models=0
        )


def test_sapientml_raise_error_if_pp_models_can_not_load():
    import os
    import pickle

    src_root = Path(__file__).parents[2]
    model_path = src_root / "sapientml" / "code_block_generator" / "prediction_based" / "models"

    test = "test"
    assert os.path.isfile(model_path / "pp_models.pkl")
    with pytest.raises(Exception):
        with open(model_path / test / "pp_models.pkl", "rb") as f:
            pickle.load(f)


def test_sapientml_raise_error_if_mp_model_1_can_not_load():
    import os
    import pickle

    src_root = Path(__file__).parents[2]
    model_path = src_root / "sapientml" / "code_block_generator" / "prediction_based" / "models"

    test = "test"
    assert os.path.isfile(model_path / "mp_model_1.pkl")
    with pytest.raises(Exception):
        with open(model_path / test / "mp_model_1.pkl", "rb") as f:
            pickle.load(f)


def test_sapientml_raise_error_if_mp_model_2_can_not_load():
    import os
    import pickle

    src_root = Path(__file__).parents[2]
    model_path = src_root / "sapientml" / "code_block_generator" / "prediction_based" / "models"

    test = "test"
    assert os.path.isfile(model_path / "mp_model_2.pkl")
    with pytest.raises(Exception):
        with open(model_path / test / "mp_model_2.pkl", "rb") as f:
            pickle.load(f)


def test_sapientml_works_with_hyperparameter_tuning_true(testdata_df_light):
    cls_ = SapientML()

    ret = cls_.generate_code(
        training_data=testdata_df_light,
        target_columns=["target_number"],
        task_type="regression",
        hyperparameter_tuning=True,
        hyperparameter_tuning_n_trials=1,
    )

    assert ret.final_script and ret.final_script[1].score


@pytest.mark.parametrize("delimiter", [" ", ";", "|"])
def test_raise_error_if_csv_delimiters_are_different(testdata_df_light, delimiter):
    with pytest.raises(Exception):
        cls_ = SapientML()
        cls_.generate_code(
            training_data=testdata_df_light,
            target_columns=["target_number"],
            task_type="regression",
            adaptation_csv_delimiter=delimiter,
        )


def test_sapientml_with_hpo_works_for_classification_task(testdata_df_light, caplog):
    logging.disable(logging.NOTSET)
    testdata_df_light = testdata_df_light[
        ["target_category_binary_num", "explanatory_number", "explanatory_multi_category_nonnum"]
    ]
    cls_ = SapientML()
    ret = cls_.generate_code(
        training_data=testdata_df_light,
        target_columns=["target_category_binary_num"],
        task_type="classification",
        n_models=16,
        hyperparameter_tuning=True,
        hyperparameter_tuning_n_trials=1,
        hyperparameter_tuning_timeout=120,
    )

    assert ret.final_script[0] and ret.final_script[1].score and "Error" not in caplog.text
    caplog.clear()
    logging.disable(logging.FATAL)


def test_sapientml_works_for_classification_with_stratification(testdata_df_light):
    cls_ = SapientML()
    ret = cls_.generate_code(
        training_data=testdata_df_light,
        target_columns=["target_category_binary_num"],
        task_type="classification",
        split_stratification=True,
        adaptation_metric="LogLoss",
    )

    assert "stratify" in ret.final_script[0].code_for_test


def test_sapientml_works_for_regression_with_stratification(testdata_df_light):
    cls_ = SapientML()
    ret = cls_.generate_code(
        training_data=testdata_df_light,
        target_columns=["target_number"],
        task_type="regression",
        split_stratification=True,
    )

    assert "stratify" not in ret.final_script[0].code_for_test
