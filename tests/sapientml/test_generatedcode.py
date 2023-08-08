import pickle
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest
from sapientml.executor import PipelineExecutor, run
from sapientml.main import SapientML

fxdir = Path("tests/fixtures").absolute()


@pytest.fixture(scope="function")
def test_data():
    return pd.read_csv(fxdir / "datasets" / "testdata_df.csv")


@pytest.fixture(scope="function")
def test_df_train():
    return pd.read_csv(fxdir / "datasets" / "testdata_train.csv")


@pytest.fixture(scope="function")
def test_df_valid():
    return pd.read_csv(fxdir / "datasets" / "testdata_valid.csv")


@pytest.fixture(scope="function")
def test_df_test():
    return pd.read_csv(fxdir / "datasets" / "testdata_test.csv")


@pytest.fixture(scope="function")
def setup_request_parameters():
    def _request_parameters():
        test_root = Path(__file__).parents[1]
        with open(test_root / "fixtures" / "params" / "task.pkl", mode="rb") as f:
            task = pickle.load(f)
        with open(test_root / "fixtures" / "params" / "config.pkl", mode="rb") as f:
            config = pickle.load(f)
        with open(test_root / "fixtures" / "params" / "dataset.pkl", mode="rb") as f:
            dataset = pickle.load(f)
        return task, config, dataset

    return _request_parameters


@pytest.fixture(scope="function")
def make_tempdir(dir=fxdir / "outputs"):
    with tempfile.TemporaryDirectory(dir=dir) as temp_dir_path_str:
        temp_dir = Path(temp_dir_path_str).absolute()
        temp_dir.mkdir(exist_ok=True)
        yield temp_dir


@pytest.fixture(scope="function")
def execute_pipeline():
    def _execute(dataset, task, config, temp_dir, initial_timeout=60):
        sml = SapientML()
        sml.dataset = dataset
        sml.task = task
        sml.config = config
        pipelines = sml._generate()

        executor = PipelineExecutor()
        pipeline_results = executor.execute(
            pipeline_list=pipelines,
            initial_timeout=initial_timeout,
            output_dir=temp_dir,
            cancel=None,
        )
        return pipeline_results

    return _execute


@pytest.fixture(scope="function")
def execute_code_for_test():
    def _execute(pipeline_results, temp_dir):
        test_result_df = pd.DataFrame(
            index=range(len(pipeline_results)), columns=["returncode", "model", "result", "code_for_test"]
        )

        save_file_path = (temp_dir / "code.py").absolute().as_posix()
        for i in range(len(pipeline_results)):
            code_for_test = pipeline_results[i][0].code_for_test
            with open(save_file_path, "w", encoding="utf-8") as f:
                f.write(code_for_test)
            test_result = run(save_file_path, 300, None)
            test_result_df.loc[i, "returncode"] = test_result.returncode
            test_result_df.loc[i, "model"] = pipeline_results[i][0].model.label_name.split(":")[-2]
            test_result_df.loc[i, "result"] = test_result
            test_result_df.loc[i, "code_for_test"] = code_for_test
        return test_result_df

    return _execute


@pytest.mark.parametrize("adaptation_metric", ["r2", "RMSE", "RMSLE", "MAE"])
@pytest.mark.parametrize("target_col", ["target_number", "target_number_large_scale_neg"])
def test_regressor(
    adaptation_metric,
    target_col,
    setup_request_parameters,
    make_tempdir,
    execute_pipeline,
    execute_code_for_test,
    test_data,
):
    task, config, dataset = setup_request_parameters()

    # test pattern setting
    df = test_data
    n_models = 14  # Maximum number of types in regressor is 14
    task.n_models = n_models
    task.task_type = "regression"
    task.adaptation_metric = adaptation_metric
    task.target_columns = [target_col]

    dataset.training_dataframe = df
    dataset.training_data_path = (fxdir / "datasets" / "testdata_df.csv").as_posix()

    temp_dir = make_tempdir
    pipeline_results = execute_pipeline(dataset, task, config, temp_dir, initial_timeout=60)
    test_result_df = execute_code_for_test(pipeline_results, temp_dir)

    for i in range(len(test_result_df)):
        model = test_result_df.loc[i, "model"]
        returncode = test_result_df.loc[i, "returncode"]
        if model == "SVR":
            # "AttributeError:var not found" occurs in SVR because of sparse_matrix
            assert returncode == 1
        else:
            assert returncode == 0


@pytest.mark.parametrize("adaptation_metric", ["r2"])
@pytest.mark.parametrize("target_col", ["target_number_large_scale"])
def test_regressor_preprocess_scaling_log(
    adaptation_metric,
    target_col,
    setup_request_parameters,
    make_tempdir,
    execute_pipeline,
    execute_code_for_test,
    test_data,
):
    task, config, dataset = setup_request_parameters()

    # test pattern setting
    df = test_data
    n_models = 1
    task.n_models = n_models
    task.task_type = "regression"
    task.adaptation_metric = adaptation_metric
    task.target_columns = [target_col]
    # Exclude columns that prevent the application of Scaling:log preprocessing
    task.ignore_columns.extend(
        ["explanatory_multi_category_num", "target_category_multi_num", "target_category_binary_num"]
    )

    dataset.training_dataframe = df
    dataset.training_data_path = (fxdir / "datasets" / "testdata_df.csv").as_posix()
    temp_dir = make_tempdir
    pipeline_results = execute_pipeline(dataset, task, config, temp_dir, initial_timeout=60)
    test_result_df = execute_code_for_test(pipeline_results, temp_dir)
    code_for_test = test_result_df.loc[0, "code_for_test"]
    if "large_scale" in target_col:
        assert "np.log" in code_for_test


@pytest.mark.parametrize("adaptation_metric", ["r2"])
@pytest.mark.parametrize(
    "target_col", ["target_number", "target_number_large_scale", "target_number_neg", "target_number_large_scale_neg"]
)
def test_regressor_notext(
    adaptation_metric,
    target_col,
    setup_request_parameters,
    make_tempdir,
    execute_pipeline,
    execute_code_for_test,
    test_data,
):
    # Test with no sparse df
    task, config, dataset = setup_request_parameters()

    # test pattern setting
    df = test_data
    n_models = 14  # Maximum number of types in regressor is 14
    task.n_models = n_models
    task.task_type = "regression"
    task.adaptation_metric = adaptation_metric
    task.target_columns = [target_col]
    task.ignore_columns.extend(
        ["explanatory_text_english", "explanatory_text_japanese", "explanatory_json", "explanatory_list"]
    )

    dataset.training_dataframe = df
    dataset.training_data_path = (fxdir / "datasets" / "testdata_df.csv").as_posix()
    # dataset.meta_features_pp["feature:str_text_presence"] = 0.0

    temp_dir = make_tempdir
    pipeline_results = execute_pipeline(dataset, task, config, temp_dir, initial_timeout=60)
    test_result_df = execute_code_for_test(pipeline_results, temp_dir)
    for i in range(len(test_result_df)):
        returncode = test_result_df.loc[i, "returncode"]
        assert returncode == 0


@pytest.mark.parametrize(
    "adaptation_metric", ["f1", "auc", "ROC_AUC", "accuracy", "Gini", "LogLoss", "MCC", "QWK", "MAP_3"]
)
@pytest.mark.parametrize(
    "target_col",
    [
        "target_category_binary_num",
        "target_category_multi_nonnum",
        "target_category_binary_boolean",
    ],
)
def test_classifier(
    adaptation_metric,
    target_col,
    setup_request_parameters,
    make_tempdir,
    execute_pipeline,
    execute_code_for_test,
    test_data,
):
    task, config, dataset = setup_request_parameters()

    # test pattern setting
    df = test_data
    n_models = 16

    task.n_models = n_models
    task.task_type = "classification"
    task.adaptation_metric = adaptation_metric
    task.target_columns = [target_col]
    if df[target_col].nunique() > 2:
        task.is_multiclass = True
    else:
        task.is_multiclass = False

    dataset.training_dataframe = df
    dataset.training_data_path = (fxdir / "datasets" / "testdata_df.csv").as_posix()

    temp_dir = make_tempdir
    pipeline_results = execute_pipeline(dataset, task, config, temp_dir, initial_timeout=60)
    test_result_df = execute_code_for_test(pipeline_results, temp_dir)
    metric_needing_predict_proba = ["Gini", "auc", "ROC_AUC", "LogLoss", "MAP_K"]
    for i in range(len(test_result_df)):
        model = test_result_df.loc[i, "model"]
        returncode = test_result_df.loc[i, "returncode"]
        if (
            adaptation_metric in metric_needing_predict_proba or adaptation_metric.startswith("MAP_")
        ) and model == "LinearSVC":
            # AttributeError: 'LinearSVC' object has no attribute 'predict_proba'
            assert returncode == 1
        elif (
            adaptation_metric in metric_needing_predict_proba or adaptation_metric.startswith("MAP_")
        ) and model == "SGDClassifier":
            # AttributeError: probability estimates are not available for loss='hinge' (‘hinge’ gives a linear SVM.)
            assert returncode == 1
        elif model == "SVC":
            # "AttributeError:var not found" occurs in SVC because of sparse_matrix
            assert returncode == 1
        elif model == "GaussianNB":
            # Sparse matrix is not supported
            assert returncode == 1
        elif model == "MultinomialNB":
            # Negative value is not supported
            assert returncode == 1
        else:
            assert returncode == 0


@pytest.mark.parametrize("adaptation_metric", ["f1"])
@pytest.mark.parametrize(
    "target_col",
    [
        "target_category_multi_num",
        "target_category_binary_nonnum",
        "target_category_binary_has_japanese",
        "target_category_binary_imbalance",
        #  "target_category_multi_imbalance",  # SMOTE is not applied to multi-class target data
    ],
)
def test_classifier_target_pattern(
    adaptation_metric,
    target_col,
    setup_request_parameters,
    make_tempdir,
    execute_pipeline,
    execute_code_for_test,
    test_data,
):
    task, config, dataset = setup_request_parameters()

    # test pattern setting
    df = test_data
    n_models = 16

    task.n_models = n_models
    task.task_type = "classification"
    task.adaptation_metric = adaptation_metric
    task.target_columns = [target_col]
    if df[target_col].nunique() > 2:
        task.is_multiclass = True
    else:
        task.is_multiclass = False

    dataset.training_dataframe = df
    dataset.training_data_path = (fxdir / "datasets" / "testdata_df.csv").as_posix()
    temp_dir = make_tempdir
    pipeline_results = execute_pipeline(dataset, task, config, temp_dir, initial_timeout=60)
    test_result_df = execute_code_for_test(pipeline_results, temp_dir)
    for i in range(len(test_result_df)):
        model = test_result_df.loc[i, "model"]
        returncode = test_result_df.loc[i, "returncode"]
        if model == "SVC":
            # "AttributeError:var not found" occurs in SVC because of sparse_matrix
            assert returncode == 1
        elif model == "GaussianNB":
            # Sparse matrix is not supported
            assert returncode == 1
        elif model == "MultinomialNB":
            # Negative value is not supported
            assert returncode == 1
        else:
            assert returncode == 0


@pytest.mark.parametrize("adaptation_metric", ["f1"])
@pytest.mark.parametrize(
    "target_col",
    [
        "target_category_binary_imbalance",
    ],
)
def test_classifier_preprocess(
    adaptation_metric,
    target_col,
    setup_request_parameters,
    make_tempdir,
    execute_pipeline,
    execute_code_for_test,
    test_data,
):
    task, config, dataset = setup_request_parameters()

    # test pattern setting
    df = test_data
    n_models = 1
    task.n_models = n_models
    task.task_type = "classification"
    task.adaptation_metric = adaptation_metric
    task.target_columns = [target_col]
    if df[target_col].nunique() > 2:
        task.is_multiclass = True
    else:
        task.is_multiclass = False

    dataset.training_dataframe = df
    dataset.training_data_path = (fxdir / "datasets" / "testdata_df.csv").as_posix()
    task.ignore_columns.extend(
        ["explanatory_multi_category_num", "target_category_multi_num", "target_category_binary_num"]
    )
    temp_dir = make_tempdir
    pipeline_results = execute_pipeline(dataset, task, config, temp_dir, initial_timeout=60)
    test_result_df = execute_code_for_test(pipeline_results, temp_dir)
    code_for_test = test_result_df.loc[0, "code_for_test"]
    assert "SMOTE" in code_for_test
    assert "StandardScaler" in code_for_test


@pytest.mark.parametrize("adaptation_metric", ["f1"])
@pytest.mark.parametrize(
    "target_col",
    [
        "target_category_binary_num",
        "target_category_multi_nonnum",
        "target_category_binary_boolean",
    ],
)
def test_classifier_notext_nonegative_explanatry(
    adaptation_metric,
    target_col,
    setup_request_parameters,
    make_tempdir,
    execute_pipeline,
    execute_code_for_test,
    test_data,
):
    task, config, dataset = setup_request_parameters()

    # test pattern setting
    df = test_data
    n_models = 16
    task.n_models = n_models
    task.task_type = "classification"
    task.adaptation_metric = adaptation_metric
    task.target_columns = [target_col]
    use_cols = [target_col] + ["explanatory_number", "explanatory_multi_category_nonnum"]
    ignore_cols = [col for col in df.columns if col not in use_cols]
    task.ignore_columns.extend(ignore_cols)
    task.ignore_columns = list(set(task.ignore_columns))

    dataset.training_dataframe = df
    dataset.training_data_path = (fxdir / "datasets" / "testdata_df.csv").as_posix()
    if "multi" in target_col:
        task.is_multiclass = True
    else:
        task.is_multiclass = False

    temp_dir = make_tempdir
    pipeline_results = execute_pipeline(dataset, task, config, temp_dir, initial_timeout=60)
    test_result_df = execute_code_for_test(pipeline_results, temp_dir)
    for i in range(len(test_result_df)):
        returncode = test_result_df.loc[i, "returncode"]
        assert returncode == 0


@pytest.mark.parametrize("adaptation_metric", ["f1", "accuracy", "MCC", "QWK"])
@pytest.mark.parametrize(
    "target_col",
    [
        "target_category_binary_num",
        "target_category_multi_nonnum",
    ],
)
def test_classifier_proba(
    adaptation_metric,
    target_col,
    setup_request_parameters,
    make_tempdir,
    execute_pipeline,
    execute_code_for_test,
    test_data,
):
    task, config, dataset = setup_request_parameters()

    # test pattern setting
    df = test_data
    n_models = 16

    task.n_models = n_models
    task.task_type = "classification"
    task.adaptation_metric = adaptation_metric
    task.target_columns = [target_col]
    config.predict_option = "probability"
    if "multi" in target_col:
        task.is_multiclass = True
    else:
        task.is_multiclass = False

    dataset.training_dataframe = df
    dataset.training_data_path = (fxdir / "datasets" / "testdata_df.csv").as_posix()

    temp_dir = make_tempdir
    pipeline_results = execute_pipeline(dataset, task, config, temp_dir, initial_timeout=60)
    test_result_df = execute_code_for_test(pipeline_results, temp_dir)
    for i in range(len(test_result_df)):
        model = test_result_df.loc[i, "model"]
        returncode = test_result_df.loc[i, "returncode"]
        if model == "LinearSVC":
            # AttributeError: 'LinearSVC' object has no attribute 'predict_proba'
            assert returncode == 1
        elif model == "SGDClassifier":
            # AttributeError: probability estimates are not available for loss='hinge' (‘hinge’ gives a linear SVM.)
            assert returncode == 1
        elif model == "SVC":
            # "AttributeError:var not found" occurs in SVC because of sparse_matrix
            assert returncode == 1
        elif model == "GaussianNB":
            # Sparse matrix is not supported
            assert returncode == 1
        elif model == "MultinomialNB":
            # Negative value is not supported
            assert returncode == 1
        else:
            assert returncode == 0


@pytest.mark.parametrize("adaptation_metric", ["r2"])
@pytest.mark.parametrize("target_col", ["target_number_large_scale", "target_number_neg"])
def test_preprocess_specify_train_valid_test(
    adaptation_metric,
    target_col,
    setup_request_parameters,
    make_tempdir,
    execute_pipeline,
    execute_code_for_test,
    test_df_train,
    test_df_valid,
    test_df_test,
):
    task, config, dataset = setup_request_parameters()

    # test pattern setting
    n_models = 3
    task.n_models = n_models
    task.task_type = "regression"
    task.adaptation_metric = adaptation_metric
    task.target_columns = [target_col]

    dataset.training_dataframe = test_df_train
    dataset.validation_dataframe = test_df_valid
    dataset.test_dataframe = test_df_test

    dataset.training_data_path = (fxdir / "datasets" / "testdata_train.csv").as_posix()
    dataset.validation_data_path = (fxdir / "datasets" / "testdata_valid.csv").as_posix()
    dataset.test_data_path = (fxdir / "datasets" / "testdata_test.csv").as_posix()

    temp_dir = make_tempdir
    pipeline_results = execute_pipeline(dataset, task, config, temp_dir, initial_timeout=60)
    test_result_df = execute_code_for_test(pipeline_results, temp_dir)

    for i in range(len(test_result_df)):
        model = test_result_df.loc[i, "model"]
        returncode = test_result_df.loc[i, "returncode"]
        code_for_test = test_result_df.loc[i, "code_for_test"]

        assert "TRAIN-TEST SPLIT" not in code_for_test
        assert "Remove special symbols" in code_for_test
        assert "HANDLE MIXED TYPE" in code_for_test
        assert "CONVERT INF TO NAN" in code_for_test
        assert "HANDLE JAPANESE TEXT" in code_for_test
        assert "DISCARD IRRELEVANT COLUMNS" in code_for_test
        assert "Preprocess:SimpleImputer" in code_for_test
        assert "_NUMERIC_COLS_WITH_MISSING_VALUES" in code_for_test
        assert "_NUMERIC_ALMOST_MISSING_COLS" in code_for_test
        assert "_STRING_COLS_WITH_MISSING_VALUES" in code_for_test
        assert "_STRING_ALMOST_MISSING_COLS" in code_for_test
        assert "Preprocess:OrdinalEncoder" in code_for_test
        assert "Preprocess:DateTime" in code_for_test
        assert "Preprocess:TextPreprocessing" in code_for_test
        assert "Preprocess:TfidfVectorizer" in code_for_test
        if model == "SVR":
            # "AttributeError:var not found" occurs in SVR because of sparse_matrix
            assert returncode == 1
        else:
            assert returncode == 0


def test_sapientml_works_initial_timeout(setup_request_parameters, make_tempdir, execute_pipeline, test_data):
    task, config, dataset = setup_request_parameters()

    # test pattern setting
    df = test_data
    n_models = 3

    task.n_models = n_models
    task.task_type = "regression"
    task.target_columns = ["target_number"]

    dataset.training_dataframe = df
    dataset.training_data_path = (fxdir / "datasets" / "testdata_df.csv").as_posix()

    temp_dir = make_tempdir
    pipeline_results = execute_pipeline(dataset, task, config, temp_dir, initial_timeout=1)
    for i in range(len(pipeline_results)):
        error = pipeline_results[i][1].error
        returncode = pipeline_results[i][1].returncode
        if error:
            assert (error == "Timeout") and (returncode == -9)


def test_timeout_works_hyperparameter_tuning_timeout(
    setup_request_parameters, make_tempdir, execute_pipeline, test_data
):
    task, config, dataset = setup_request_parameters()

    # test pattern setting
    df = test_data
    n_models = 1

    task.n_models = n_models
    task.task_type = "regression"
    task.target_columns = ["target_number"]

    config.hyperparameter_tuning = True
    if config.hyperparameter_tuning:
        initial_timeout = 0
    config.hyperparameter_tuning_n_trials = 10
    config.hyperparameter_tuning_timeout = 1

    dataset.training_dataframe = df
    dataset.training_data_path = (fxdir / "datasets" / "testdata_df.csv").as_posix()

    temp_dir = make_tempdir
    start_time = time.time()
    pipeline_results = execute_pipeline(dataset, task, config, temp_dir, initial_timeout=initial_timeout)
    end_time = time.time()
    process_time = end_time - start_time

    assert pipeline_results[0][1].returncode == 0
    assert process_time < 20
