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

import collections
import warnings
from typing import Literal, Union

import numpy
import pandas as pd
import scipy
import scipy.stats
from sklearn.preprocessing import StandardScaler

from . import ps_macros
from .design import search_space

MetaFeatures = dict[str, Union[float, int, str, list[str], None]]

ban_features = [
    "feature:not_basic_cols",
    "feature:dominant_0.8",
    "feature:missing_10-50_cols",
    "feature:missing_50-90_cols",
    "feature:missing_90-100_cols",
    "feature:negative_cols",
    "feature:median_0-1_cols",
    "feature:median_1-10_cols",
    "feature:median_10-100_cols",
    "feature:median_100-1000_cols",
    "feature:median_1000-10000_cols",
    "feature:median_10000_cols",
    "feature:iqr_0-1_cols",
    "feature:iqr_1-10_cols",
    "feature:iqr_10-100_cols",
    "feature:iqr_100-1000_cols",
    "feature:iqr_1000-10000_cols",
    "feature:iqr_10000_cols",
    "feature:str_date",
    "feature:str_catg",
    "feature:str_num",
    "feature:str_other",
    "feature:missing_values_presence",
    "feature:missing_0-10_cols",
    "feature:target_str_catg",
    "feature:target_str_text",
    "feature:target_str_other",
    "feature:target_catg_num_max",
    "feature:target_catg_num_min",
    "feature:ttest_max",
    "feature:ttest_min",
    "feature:kstest_max",
    "feature:kstest_min",
    "feature:pearsonr_corr_max",
    "feature:target_kurtosis_max",
    "feature:target_kurtosis_min",
    "feature:missing_values_special",
    "feature:missing_values_rows",
    "feature:missing_values_named_columns",
    "feature:avg_num_words",
    "feature:num_unique_words",
    "feature:str_catg_small",
    "feature:str_catg_large",
    "feature:num_catg_binary",
    "feature:num_catg_small",
    "feature:num_catg_large",
    "feature:pearsonr_p_min",
    "feature:missing_values_special",
]

normalize_features = [
    "feature:str_date",
    "feature:str_catg",
    "feature:str_num",
    "feature:str_text",
    "feature:str_other",
    "feature:num_catg",
    "feature:num_cont",
    "feature:str_catg_binary",
    "feature:str_catg_small",
    "feature:str_catg_large",
    "feature:num_catg_binary",
    "feature:num_catg_small",
    "feature:num_catg_large",
    "feature:outlier_cols",
    "feature:many_outlier_cols",
    "feature:kurtosis_normal",
    "feature:kurtosis_uniform",
    "feature:kurtosis_tailed",
    "feature:dist_normal",
    "feature:dist_uniform",
    "feature:dist_poisson",
    "feature:dominant_0.9",
    "correlated_cols",
    "feature:ttest_max",
    "feature:ttest_min",
    "feature:kstest_max",
    "feature:kstest_min",
    "feature:pearsonr_corr_max",
    "feature:pearsonr_corr_min",
    "feature:pearsonr_p_max",
    "feature:pearsonr_p_min",
    "feature:missing_values_presence",
    "feature:missing_0-10_cols",
    "feature:missing_values_named_columns",
]

transform_features = {
    "feature:num_of_rows": lambda x: numpy.log1p(x),
    # 'feature:num_of_features': lambda x: np.log1p(x),
    # 'feature:missing_values_rows': lambda x: np.log1p(x)
}


def _is_real_dtype(column):
    return (
        pd.api.types.is_numeric_dtype(column)
        and (not pd.api.types.is_complex_dtype(column))
        and (not pd.api.types.is_bool_dtype(column))
    )


def _is_realbool_dtype(column):
    return pd.api.types.is_numeric_dtype(column) and (not pd.api.types.is_complex_dtype(column))


def real_feature_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    nf = "feature:num_of_features"
    for col in normalize_features:
        if nf in df.columns:
            if col in df.columns:
                df.loc[:, col] = df[col].astype(float) / df[nf].astype(float)

    for key in transform_features:
        if key in df.columns:
            df.loc[:, key] = transform_features[key](df[key].astype(float))

    valid_cols = [col for col in df.columns if col not in ban_features]
    df = df[valid_cols]
    return df


def test_feature_preprocess(raw_df: pd.DataFrame, is_clf_task: Literal[0, 1]):
    raw_df = raw_df.drop(axis=1, columns=["target_column_name"]).reset_index(drop=True)
    clf_row = [is_clf_task]
    raw_df["feature:is_cls"] = clf_row
    return real_feature_preprocess(raw_df)


# Generate the model mata features for online and offline processing
def _collect_model_meta_features(data_df: pd.DataFrame, target_column_name: Union[str, list[str]]):
    if isinstance(target_column_name, str):
        target_column_name = [target_column_name]
    try:
        dataX = data_df.drop(target_column_name, axis=1)
        datay = data_df[target_column_name]
    except Exception:
        raise RuntimeError(f"No target column {target_column_name} found")

    meta_feature_dict: MetaFeatures = collections.OrderedDict({})

    feature_dict: MetaFeatures = collections.OrderedDict()
    feature_dict["num_of_rows"] = dataX.shape[0]
    feature_dict["num_of_features"] = dataX.shape[1]

    if dataX.shape[0] > 100000:
        indx = numpy.random.choice(dataX.shape[0], 100000, replace=False)
        dataX = dataX.iloc[indx]
        datay = datay.iloc[indx]

    sampledX = dataX.select_dtypes("object")
    is_basic_type = sampledX.applymap(
        lambda x: isinstance(x, int) or isinstance(x, float) or isinstance(x, bool) or isinstance(x, str)
    )
    not_basic_cols = sampledX.columns[(~is_basic_type).any()]
    feature_dict["not_basic_cols"] = not_basic_cols.shape[0]
    for col in not_basic_cols:
        dataX[col] = dataX[col].astype(str)

    column_type_dict = _collect_csv_column_type_presence_pp(dataX, preprocess=False)
    feature_dict.update(column_type_dict)

    feature_dict.update(_collect_csv_column_with_outlier(dataX))

    feature_dict.update(_collect_csv_column_kurtosis(dataX))

    feature_dict.update(_collect_csv_column_num_distribution(dataX))

    feature_dict.update(_collect_csv_column_dominant(dataX))

    feature_dict["correlated_cols"] = _collect_csv_column_correlated(dataX)

    num_of_missing_value_cols = _collect_csv_missing_value_presence(dataX)
    nan_0_10, nan_10_50, nan_50_90, nan_90_100 = _collect_csv_missing_value_rate_group(dataX)
    (
        neg,
        med_0_1,
        med_1_10,
        med_10_100,
        med_100_1000,
        med_1000_10000,
        med_10000,
        iqr_0_1,
        iqr_1_10,
        iqr_10_100,
        iqr_100_1000,
        iqr_1000_10000,
        iqr_10000,
    ) = _collect_csv_value_range_group(dataX)

    feature_dict["sparseness"] = _calc_sparseness(dataX)

    feature_dict["missing_values_presence"] = str(num_of_missing_value_cols)
    feature_dict["missing_0-10_cols"] = str(nan_0_10)
    feature_dict["missing_10-50_cols"] = str(nan_10_50)
    feature_dict["missing_50-90_cols"] = str(nan_50_90)
    feature_dict["missing_90-100_cols"] = str(nan_90_100)
    feature_dict["negative_cols"] = str(neg)
    feature_dict["median_0-1_cols"] = str(med_0_1)
    feature_dict["median_1-10_cols"] = str(med_1_10)
    feature_dict["median_10-100_cols"] = str(med_10_100)
    feature_dict["median_100-1000_cols"] = str(med_100_1000)
    feature_dict["median_1000-10000_cols"] = str(med_1000_10000)
    feature_dict["median_10000_cols"] = str(med_10000)
    feature_dict["iqr_0-1_cols"] = str(iqr_0_1)
    feature_dict["iqr_1-10_cols"] = str(iqr_1_10)
    feature_dict["iqr_10-100_cols"] = str(iqr_10_100)
    feature_dict["iqr_100-1000_cols"] = str(iqr_100_1000)
    feature_dict["iqr_1000-10000_cols"] = str(iqr_1000_10000)
    feature_dict["iqr_10000_cols"] = str(iqr_10000)

    feature_dict["num_target"] = datay.shape[1]

    column_types = _get_target_column_type_pp(datay, preprocess=False)
    feature_dict.update(_count_column_type_pp(column_types))

    catg_target_columns = [k for k, v in column_types.items() if "catg" in v[0]]
    feature_dict.update(_collect_csv_column_with_ttest(dataX, datay[catg_target_columns]))
    feature_dict.update(_collect_csv_column_with_kstest(dataX, datay[catg_target_columns]))

    num_target_columns = [k for k, v in column_types.items() if "num" in v[0]]
    feature_dict.update(_collect_csv_column_with_pearsonr(dataX, datay[num_target_columns]))

    small_catg_target_columns = [k for k, v in column_types.items() if "catg" in v[0] and v[1] <= 20]
    feature_dict["imbalance"] = _get_target_imbalance(datay[small_catg_target_columns])

    num_cont_target_columns = [
        k for k, v in column_types.items() if v[0] == "num_cont" or (v[0] == "num_catg" and v[1] > 20)
    ]
    target_dist_num = _collect_csv_column_num_distribution(datay[num_cont_target_columns])
    target_dist_num = collections.OrderedDict([("target_" + k, v) for k, v in target_dist_num.items()])
    feature_dict.update(target_dist_num)

    if num_cont_target_columns:
        num_y = datay[num_cont_target_columns]
        kur = num_y.kurtosis()
        feature_dict["target_kurtosis_max"] = kur.max()
        feature_dict["target_kurtosis_min"] = kur.min()
    else:
        feature_dict["target_kurtosis_max"] = None
        feature_dict["target_kurtosis_min"] = None

    feature_dict["missing_values_special"] = _get_missing_values_special(dataX)
    feature_dict["missing_values_rows"] = _get_missing_values_rows(dataX)
    feature_dict["missing_values_named_columns"] = _get_missing_values_named_columns(dataX)
    avg_num_words, num_unique_words = _get_word_count(dataX)
    feature_dict["avg_num_words"] = avg_num_words
    feature_dict["num_unique_words"] = num_unique_words

    feature_dict = collections.OrderedDict([("feature:" + k, v) for k, v in feature_dict.items()])

    meta_feature_dict["target_column_name"] = target_column_name

    meta_feature_dict.update(feature_dict)
    return meta_feature_dict


def collect_labels(annotated_notebooks_path):
    # Read the annotated notebooks and group by file name
    data = pd.read_csv(annotated_notebooks_path, encoding="utf-8")
    groups = data[["file_name", "new_label"]].groupby(["file_name"])

    # One-hot encode the labels
    feature_map_list = []
    for file_name, group in groups:
        labels = group["new_label"].tolist()

        feature_map = {}
        feature_map["file_name"] = str(file_name)
        for label in labels:
            feature_map[label] = 1
        feature_map_list.append(feature_map)

    df = pd.DataFrame(feature_map_list)
    column_with_missing_values = df.columns[df.isnull().any()].tolist()
    df[column_with_missing_values] = df[column_with_missing_values].fillna(0).astype(int)

    return df


def _is_category_column(c):
    c2 = c.loc[c.notnull()]
    num = c2.nunique()
    if num <= 2:
        return "binary"
    elif num <= 20 and (num / c2.shape[0] < 0.3):
        return "small"
    elif c2.value_counts(normalize=True).head(20).sum() > 0.8:
        return "large"
    else:
        return False


def _get_word_count(X):
    from collections import Counter

    avg_num_words = []
    num_unique_words = []
    for _, column in X.items():
        column = column.loc[column.notnull()]
        if column.shape[0] > 1000000:
            column = column.sample(1000000, random_state=17)
        try:
            if pd.api.types.is_object_dtype(column):
                results = Counter()
                column.str.lower().str.split().apply(results.update)
                num_words = sum(results.values())
                avg_num_words_ = num_words / column.shape[0]
                num_unique_words_ = len(list(results))
                avg_num_words.append(avg_num_words_)
                num_unique_words.append(num_unique_words_)
        except Exception:
            pass
    if len(avg_num_words) > 0 and len(num_unique_words) > 0:
        return (max(avg_num_words), max(num_unique_words))
    else:
        return (0, 0)


def _get_target_imbalance(Y):
    imb_max = None
    for _, yc in Y.items():
        y0 = yc.loc[yc.notnull()]
        if y0.shape[0] == 0:
            return None
        vc = y0.value_counts()
        if vc.shape[0] == 1:
            return None
        vfirst = vc.iloc[0]
        vlast = vc.iloc[-1]
        imb = 1 - vlast / vfirst
        if imb_max is None or imb_max < imb:
            imb_max = imb
    return imb_max


def _collect_csv_column_with_outlier(X):
    num_cols = [k for k, v in X.dtypes.items() if _is_realbool_dtype(v)]
    res = collections.OrderedDict()
    res["outlier_cols"] = 0
    res["many_outlier_cols"] = 0
    if len(num_cols) == 0:
        return res
    tmpX = X[num_cols]
    scaler = StandardScaler()
    import numpy as np
    from scipy import sparse

    FLOAT_DTYPES = (np.float64, np.float32, np.float16)
    first_call = not hasattr(scaler, "n_samples_seen_")
    tmpX = scaler._validate_data(
        tmpX,
        accept_sparse=("csr", "csc"),
        estimator=scaler,
        dtype=FLOAT_DTYPES,
        force_all_finite="allow-nan",
        reset=first_call,
    )
    if sparse.issparse(tmpX):
        tmpX = tmpX.toarray()
    tmpX = scaler.fit_transform(tmpX)
    tmpX[numpy.isnan(tmpX)] = 0
    bool_mat = tmpX > 3
    cols = bool_mat.any(axis=0).sum()
    res["outlier_cols"] = int(cols)
    rates = bool_mat.sum(axis=0) / bool_mat.shape[0]
    many_cols = (rates >= 0.05).sum()
    res["many_outlier_cols"] = int(many_cols)
    return res


def _get_ttest_pvalue(c, y):
    cnotnan = c.notnull()
    ynotnan = y.notnull()
    notnan = cnotnan & ynotnan
    c0 = c.loc[notnan]
    y0 = y.loc[notnan]
    vc = y0.value_counts()
    if vc.shape[0] >= 2:
        v1, v2 = vc.index[:2]
    else:
        return 1.0
    b1 = y == v1
    b2 = y == v2
    c1 = c0.loc[b1]
    c2 = c0.loc[b2]
    _, p = scipy.stats.ttest_ind(c1, c2, equal_var=False)
    return p


def _get_kstest_pvalue(c, y):
    cnotnan = c.notnull()
    ynotnan = y.notnull()
    notnan = cnotnan & ynotnan
    c0 = c.loc[notnan]
    y0 = y.loc[notnan]
    vc = y0.value_counts()
    if vc.shape[0] >= 2:
        v1, v2 = vc.index[:2]
    else:
        return 1.0
    b1 = y == v1
    b2 = y == v2
    c1 = c0.loc[b1]
    c2 = c0.loc[b2]
    _, p = scipy.stats.kstest(c1, c2)
    return p


def _get_pearsonr_values(c, y):
    cnotnan = c.notnull()
    ynotnan = y.notnull()
    notnan = cnotnan & ynotnan
    c0 = c.loc[notnan]
    y0 = y.loc[notnan]
    corr, p = scipy.stats.pearsonr(c0, y0)
    return corr, p


def _collect_csv_column_with_ttest(X, Y):
    num_cols = [k for k, v in X.dtypes.items() if _is_realbool_dtype(v)]
    cnt_max = None
    cnt_min = None
    for yn, yc in Y.items():
        cnt = 0
        for n, column in X[num_cols].items():
            p = _get_ttest_pvalue(column, yc)
            if p < 0.01:
                cnt += 1
        if cnt_max is None or cnt_max < cnt:
            cnt_max = cnt
        if cnt_min is None or cnt_min > cnt:
            cnt_min = cnt
    res = collections.OrderedDict([("ttest_max", cnt_max), ("ttest_min", cnt_min)])
    return res


def _collect_csv_column_with_kstest(X, Y):
    num_cols = [k for k, v in X.dtypes.items() if _is_realbool_dtype(v)]
    cnt_max = None
    cnt_min = None
    for _, yc in Y.items():
        cnt = 0
        for _, column in X[num_cols].items():
            p = _get_kstest_pvalue(column, yc)
            if p < 0.01:
                cnt += 1
        if cnt_max is None or cnt_max < cnt:
            cnt_max = cnt
        if cnt_min is None or cnt_min > cnt:
            cnt_min = cnt
    res = collections.OrderedDict([("kstest_max", cnt_max), ("kstest_min", cnt_min)])
    return res


def _collect_csv_column_with_pearsonr(X, Y):
    num_cols = [k for k, v in X.dtypes.items() if _is_realbool_dtype(v)]
    corr_max = None
    corr_min = None
    p_max = None
    p_min = None
    for _, yc in Y.items():
        corr_n = 0
        p_n = 0
        for _, column in X[num_cols].items():
            try:
                corr, p = _get_pearsonr_values(column, yc)
                if abs(corr) > 0.6:
                    corr_n += 1
                if p < 0.01:
                    p_n += 1
            except ValueError:
                pass
        if corr_max is None or corr_max < corr_n:
            corr_max = corr_n
        if corr_min is None or corr_min > corr_n:
            corr_min = corr_n
        if p_max is None or p_max < p_n:
            p_max = p_n
        if p_min is None or p_min > p_n:
            p_min = p_n
    res = collections.OrderedDict(
        [
            ("pearsonr_corr_max", corr_max),
            ("pearsonr_corr_min", corr_min),
            ("pearsonr_p_max", p_max),
            ("pearsonr_p_min", p_min),
        ]
    )
    return res


def _get_num_dist_pvalues(c):
    c0 = c.loc[c.notnull()]
    if c0.shape[0] == 0:
        return None
    mlen = 1000
    if c0.shape[0] > mlen:
        c0 = c0.sample(mlen, random_state=17)
    c0 = c0.astype(float)
    c0mean = c0.mean()
    c0std = c0.std()
    c0max = c0.max()
    c0min = c0.min()
    _, p_normal = scipy.stats.kstest(c0, "norm", [c0mean, c0std])
    _, p_uniform = scipy.stats.kstest(c0, "uniform", [c0min, c0max - c0min])
    _, p_poisson = scipy.stats.kstest(c0, "poisson", [c0mean])
    return p_normal, p_uniform, p_poisson


def _collect_csv_column_num_distribution(X):
    res: MetaFeatures = collections.OrderedDict([("dist_normal", 0), ("dist_uniform", 0), ("dist_poisson", 0)])
    for _, column in X.items():
        if not _is_real_dtype(column):
            continue
        ctg = _is_category_column(column)
        if not (not ctg or ctg == "large"):
            continue
        vs = _get_num_dist_pvalues(column)
        if vs is None:
            continue
        if vs[0] > 0.01:
            res["dist_normal"] += 1
        if vs[1] > 0.01:
            res["dist_uniform"] += 1
        if vs[2] > 0.01:
            res["dist_poisson"] += 1
    return res


def _collect_csv_column_kurtosis(X):
    res = collections.OrderedDict([("kurtosis_normal", 0), ("kurtosis_uniform", 0), ("kurtosis_tailed", 0)])
    for _, column in X.items():
        if not _is_real_dtype(column):
            continue
        ctg = _is_category_column(column)
        if not (not ctg or ctg == "large"):
            continue
        c0 = column.loc[column.notnull()]
        kur = c0.kurtosis()
        if abs(kur) < 0.5:
            res["kurtosis_normal"] += 1
        elif kur < -1.0:
            res["kurtosis_uniform"] += 1
        elif kur >= 0.5:
            res["kurtosis_tailed"] += 1
    return res


def _collect_csv_column_correlated(X):
    num_cols = [k for k, v in X.dtypes.items() if _is_realbool_dtype(v)]
    X0 = X[num_cols]
    if X0.shape[0] == 0:
        return 0
    max_rows = 1000
    max_cols = 1000
    if X0.shape[0] > max_rows:
        X0 = X0.sample(max_rows, random_state=17)
    if X0.shape[1] > max_cols:
        X0 = X0.sample(max_cols, axis=1, random_state=31)
    if X0.notnull().sum().sum() == 0:
        return 0
    corr = X0.corr()
    if corr.shape[0] == 0:
        return 0
    val = (corr > 0.9).sum().max() - 1
    return val  # -1 means self


def _calc_sparseness(X):
    num_empty = ((X.isnull()) | (X == 0)).sum().sum()
    num_size = X.shape[0] * X.shape[1]
    return num_empty / num_size


def _collect_csv_column_dominant(X):
    res = collections.OrderedDict([("dominant_0.8", 0), ("dominant_0.9", 0)])
    for _, c in X.items():
        cnt = c.value_counts(normalize=True, dropna=False)
        if cnt.iloc[0] > 0.8:
            res["dominant_0.8"] += 1
        if cnt.iloc[0] > 0.9:
            res["dominant_0.9"] += 1
    return res


def _collect_csv_missing_value_presence(dataset):
    num_of_missing_value_cols = len(dataset.columns[dataset.isnull().any()].tolist())
    return num_of_missing_value_cols


def _collect_csv_missing_value_rate_group(dataset):
    nan_rate = dataset.isnull().sum() / dataset.shape[0]
    cols_nan_less_than_10_percent = ((nan_rate < 0.1) & (nan_rate > 0.0)).sum()
    cols_nan_10_50_percent = ((nan_rate >= 0.1) & (nan_rate < 0.5)).sum()
    cols_nan_50_90_percent = ((nan_rate >= 0.5) & (nan_rate <= 0.9)).sum()
    cols_nan_more_than_90_percent = (nan_rate > 0.9).sum()
    return cols_nan_less_than_10_percent, cols_nan_10_50_percent, cols_nan_50_90_percent, cols_nan_more_than_90_percent


def _collect_csv_value_range_group(dataset):
    # If _is_realbool_dtype, '-' operation of iqr_s fails.
    numeric_cols = [c for c, v in dataset.dtypes.items() if _is_real_dtype(v)]
    negative_cols = (dataset[numeric_cols] < 0).any().sum()
    median_s = dataset[numeric_cols].median().abs()
    median_0_1 = (median_s <= 1).sum()
    median_1_10 = ((median_s > 1) & (median_s <= 10)).sum()
    median_10_100 = ((median_s > 10) & (median_s <= 100)).sum()
    median_100_1000 = ((median_s > 100) & (median_s <= 1000)).sum()
    median_1000_10000 = ((median_s > 1000) & (median_s <= 10000)).sum()
    median_10000 = (median_s >= 10000).sum()
    iqr_s = dataset[numeric_cols].quantile(0.75) - dataset[numeric_cols].quantile(0.25)
    iqr_0_1 = (iqr_s <= 1).sum()
    iqr_1_10 = ((iqr_s > 1) & (iqr_s <= 10)).sum()
    iqr_10_100 = ((iqr_s > 10) & (iqr_s <= 100)).sum()
    iqr_100_1000 = ((iqr_s > 100) & (iqr_s <= 1000)).sum()
    iqr_1000_10000 = ((iqr_s > 1000) & (iqr_s <= 10000)).sum()
    iqr_10000 = (iqr_s >= 10000).sum()
    return (
        negative_cols,
        median_0_1,
        median_1_10,
        median_10_100,
        median_100_1000,
        median_1000_10000,
        median_10000,
        iqr_0_1,
        iqr_1_10,
        iqr_10_100,
        iqr_100_1000,
        iqr_1000_10000,
        iqr_10000,
    )


def _get_missing_values_special(dataset):
    feature = dataset.isin(["?", "??", "-", "--", "---"]).any().sum()
    return int(feature)


def _get_missing_values_rows(dataset):
    feature = dataset.isnull().any(axis=1).sum()
    return int(feature)


def _get_missing_values_named_columns(dataset):
    unnamed = dataset.columns.str.match("Unnamed").sum()
    feature = len(dataset.columns) - unnamed
    return int(feature)


# Generate the model meta_feature for offline processing
def compute_model_meta_features(df, proj_name, project, target_column_name):
    try:
        meta_feature_dict = _collect_model_meta_features(df, target_column_name)
    except Exception as e:
        print("Could not generate model meta-features for {}".format(proj_name))
        print("Exception: {}".format(e))
        # raise
        return None

    # Add the file name and the notebook name
    meta_feature_dict["file_name"] = project.file_name
    meta_feature_dict["notebook_name"] = project.notebook_name
    meta_feature_dict["project_name"] = project.project_name
    meta_feature_dict["accuracy"] = project.accuracy
    meta_feature_dict["csv_name"] = project.csv_name

    # Move the notebook name to the second item in the ordered dict
    meta_feature_dict.move_to_end("accuracy", last=False)
    meta_feature_dict.move_to_end("csv_name", last=False)
    meta_feature_dict.move_to_end("notebook_name", last=False)
    meta_feature_dict.move_to_end("project_name", last=False)
    meta_feature_dict.move_to_end("file_name", last=False)

    return meta_feature_dict


# Generate the pp mata features for online and offline processing
def _collect_pp_meta_features(data_df: pd.DataFrame, target_column_name: Union[str, list[str]]):
    if isinstance(target_column_name, str):
        target_column_name = [target_column_name]
    try:
        dataX = data_df.drop(target_column_name, axis=1)
        datay = data_df[target_column_name]
    except Exception:
        raise RuntimeError(f"No target column {target_column_name} found")

    meta_feature_dict: MetaFeatures = collections.OrderedDict({})

    feature_dict = collections.OrderedDict()
    feature_dict["num_of_rows"] = dataX.shape[0]
    feature_dict["num_of_features"] = dataX.shape[1]

    if dataX.shape[0] > 1000000:
        indx = numpy.random.choice(dataX.shape[0], 1000000, replace=False)
        dataX = dataX.iloc[indx]
        datay = datay.iloc[indx]
    column_type_dict = _collect_csv_column_type_presence_pp(dataX, preprocess=True)
    feature_dict.update(column_type_dict)
    feature_dict["missing_values_presence"] = _get_missing_values_presence(dataX)
    feature_dict["all_missing_values"] = _collect_csv_missing_all_value_presence(dataX)
    feature_dict["num_target"] = datay.shape[1]
    column_types = _get_target_column_type_pp(datay, preprocess=True)
    feature_dict.update(_count_column_type_pp(column_types))
    feature_dict["max_normalized_mean"] = _get_max_normalized_mean(dataX)
    feature_dict["max_normalized_stddev"] = _get_max_normalized_stddev(dataX)
    feature_dict["normalized_variation_across_columns"] = _get_normalized_variation_across_columns(dataX)
    feature_dict["max_skewness"] = _get_max_skewness(data_df)
    feature_dict["target_imbalance_score"] = _get_target_imbalance_score(datay)
    feature_dict = collections.OrderedDict([("feature:" + k, v) for k, v in feature_dict.items()])
    meta_feature_dict["target_column_name"] = target_column_name
    meta_feature_dict.update(feature_dict)
    return meta_feature_dict


def _collect_column_meta_features(column_df: pd.DataFrame):
    meta_feature_dict: MetaFeatures = collections.OrderedDict({})
    feature_dict: MetaFeatures = collections.OrderedDict()
    column_type_dict = _collect_csv_column_type_presence_pp(column_df, preprocess=True)
    feature_dict.update(column_type_dict)
    feature_dict["missing_values_presence"] = _get_missing_values_presence(column_df)
    feature_dict["all_missing_values"] = _collect_csv_missing_all_value_presence(column_df)
    feature_dict["max_normalized_mean"] = _get_max_normalized_mean(column_df)
    feature_dict["max_normalized_stddev"] = _get_max_normalized_stddev(column_df)
    feature_dict["normalized_variation_across_columns"] = _get_normalized_variation_across_columns(column_df)
    feature_dict["max_skewness"] = _get_max_skewness(column_df)
    feature_dict = collections.OrderedDict([("feature:" + k, v) for k, v in feature_dict.items()])
    meta_feature_dict.update(feature_dict)
    return meta_feature_dict


def _collect_csv_column_type_presence_pp(X, preprocess):
    res: MetaFeatures = collections.OrderedDict()
    types = [
        "str_date",
        "str_catg",
        "str_num",
        "str_text",
        "str_other",
        "num_catg",
        "num_cont",
        "str_catg_binary",
        "str_catg_small",
        "str_catg_large",
        "num_catg_binary",
        "num_catg_small",
        "num_catg_large",
    ]
    for t in types:
        res[t] = 0
    if preprocess:
        res["str_other"] = []
    for name, column in X.items():
        if pd.api.types.is_object_dtype(column):
            if _is_date_column_pp(column):
                res["str_date"] += 1
            else:
                if preprocess:
                    catg_type = _is_category_column_pp(column)
                else:
                    catg_type = _is_category_column(column)
                if catg_type:
                    res["str_catg"] += 1
                    cellname = "str_catg_%s" % (catg_type,)
                    res[cellname] += 1
                elif _is_strnum_column_pp(column):
                    res["str_num"] += 1
                elif _is_text_column_pp(column, preprocess=True):
                    res["str_text"] += 1
                else:
                    if preprocess:
                        res["str_other"].append(name)
                    else:
                        res["str_other"] += 1
        elif _is_realbool_dtype(column):
            if preprocess:
                catg_type = _is_category_column_pp(column)
            else:
                catg_type = _is_category_column(column)
            if catg_type:
                res["num_catg"] += 1
                cellname = "num_catg_%s" % (catg_type,)
                res[cellname] += 1
            else:
                res["num_cont"] += 1
        elif pd.api.types.is_datetime64_any_dtype(column):
            res["str_date"] += 1
        else:
            raise RuntimeError()
    if preprocess:
        res["str_category_presence"] = 1 if res["str_catg"] > 0 else 0
        res["str_category_binary_presence"] = 1 if res["str_catg_binary"] > 0 else 0
        res["str_category_small_presence"] = 1 if res["str_catg_small"] > 0 else 0
        res["str_category_large_presence"] = 1 if res["str_catg_large"] > 0 else 0
        res["str_text_presence"] = 1 if res["str_text"] > 0 else 0
        res["str_date_presence"] = 1 if res["str_date"] > 0 else 0
    return res


def _get_target_column_type_pp(Y, preprocess):
    res = collections.OrderedDict()
    for (
        name,
        column,
    ) in Y.items():
        if pd.api.types.is_object_dtype(column):
            if preprocess:
                catg_type = _is_category_column_pp(column)
            else:
                catg_type = _is_category_column(column)
            if catg_type:
                catg_num = column.loc[column.notnull()].nunique()
                res[name] = ("str_catg", catg_num)
            elif _is_text_column_pp(column, preprocess=preprocess):
                res[name] = ("str_text",)
            else:
                res[name] = ("str_other",)
        elif _is_realbool_dtype(column):
            if preprocess:
                catg_type = _is_category_column_pp(column)
            else:
                catg_type = _is_category_column(column)
            if catg_type:
                catg_num = column.loc[column.notnull()].nunique()
                res[name] = ("num_catg", catg_num)
            else:
                res[name] = ("num_cont",)
        else:
            raise RuntimeError(column.dtype.name)
    return res


def _count_column_type_pp(column_type):
    res = collections.OrderedDict()
    types = ["str_catg", "str_text", "str_other", "num_catg", "num_cont"]
    catg_num_max = numpy.nan
    catg_num_min = numpy.nan
    for t in types:
        res["target_" + t] = 0
    for _, v in column_type.items():
        res["target_" + v[0]] += 1
        if "catg" in v[0]:
            if numpy.isnan(catg_num_max) or catg_num_max < v[1]:
                catg_num_max = v[1]
            if numpy.isnan(catg_num_min) or catg_num_min > v[1]:
                catg_num_min = v[1]
    res["target_catg_num_max"] = catg_num_max
    res["target_catg_num_min"] = catg_num_min
    return res


def _collect_csv_missing_all_value_presence(dataset):
    cols_with_all_missing_values = dataset.columns[dataset.isnull().all()].tolist()
    return cols_with_all_missing_values


def _get_missing_values_presence(dataset):
    num_of_missing_value_cols = _collect_csv_missing_value_presence(dataset)
    return 1 if num_of_missing_value_cols > 0 else 0


def _get_max_skewness(X):
    skewness = []
    for i in X.columns:
        if _is_real_dtype(X[i]):
            skew = abs(scipy.stats.skew(X[i].fillna(X[i].mean())))
            skewness.append(skew)

    if len(skewness) == 0:
        return 0
    else:
        max_skew = max(skewness)
        return max_skew


def _get_max_normalized_mean(X):
    numeric_cols = []
    for i in X.columns:
        if _is_real_dtype(X[i]):
            numeric_cols.append(i)
    if len(numeric_cols) == 0:
        return 0
    normalized_means = []
    for col in numeric_cols:
        c = X[col].loc[X[col].notnull()]
        if (c.max() - c.min()) != 0:
            temp = (c - c.min()) / (c.max() - c.min())
            normalized_mean = temp.mean()
            normalized_means.append(normalized_mean)
    if len(normalized_means) == 0:
        return 0
    return max(normalized_means)


def _get_normalized_variation_across_columns(X):
    numeric_cols = []
    for i in X.columns:
        if _is_real_dtype(X[i]):
            numeric_cols.append(i)
    if len(numeric_cols) == 0:
        return 0
    max_numbers = []
    for col in numeric_cols:
        c = X[col].loc[X[col].notnull()]
        if (c.max() - c.min()) != 0:
            max_numbers.append(c.abs().max())
    if len(max_numbers) == 0:
        return 0
    variation = (max(max_numbers) - min(max_numbers)) / max(max_numbers)
    return variation


def _get_max_normalized_stddev(X):
    numeric_cols = []
    for i in X.columns:
        if _is_real_dtype(X[i]):
            numeric_cols.append(i)
    if len(numeric_cols) == 0:
        return 0
    normalized_std_devs = []
    for col in numeric_cols:
        c = X[col].loc[X[col].notnull()]
        if (c.max() - c.min()) != 0:
            temp = (c - c.min()) / (c.max() - c.min())
            normalized_std_dev = numpy.std(temp)
            normalized_std_devs.append(normalized_std_dev)
    if len(normalized_std_devs) == 0:
        return 0
    return max(normalized_std_devs)


def _get_target_imbalance_score(Y):
    imb_max = numpy.nan
    for _, yc in Y.items():
        y0 = yc.loc[yc.notnull()]
        if y0.shape[0] == 0:
            return 0
        vc = y0.value_counts()
        # if there are more than 10 categories, probably it is a regression problem
        if vc.shape[0] > 2:
            return 0
        vfirst = vc.iloc[0]
        vlast = vc.iloc[-1]
        imb = 1 - vlast / vfirst
        if numpy.isnan(imb_max) or imb_max < imb:
            imb_max = imb
    return imb_max


def _is_date_column_pp(c):
    if numpy.issubdtype(c.dtype, numpy.datetime64):
        return True
    c2 = c.loc[c.notnull()]
    if c2.shape[0] > 1000:
        c2 = c2.sample(1000, random_state=17)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c2 = pd.to_datetime(c2, errors="coerce")
    ratio = c2.notnull().sum() / c2.shape[0]
    return ratio > 0.8


def _is_category_column_pp(c):
    c2 = c.loc[c.notnull()]
    num = c2.nunique()
    ratio = c2.nunique() / c2.count()
    if num == 2:
        return "binary"
    elif 2 < num <= 5:
        return "small"
    elif ratio < 0.05 or c2.value_counts(normalize=True).head(10).sum() > 0.8:
        return "large"
    else:
        return False


def _is_strnum_column_pp(c):
    c2 = c.loc[c.notnull()]
    c2 = pd.to_numeric(c2, errors="coerce")
    ratio = c2.notnull().sum() / c2.shape[0]
    return ratio > 0.9


def _is_text_column_pp(c, preprocess):
    c2 = c.loc[c.notnull()]
    if c2.shape[0] > 1000:
        c2 = c2.sample(1000, random_state=17)
    space = c2.apply(lambda x: x.count(" "))
    num = (space >= 1).sum()
    ratio_having_space = num / c2.shape[0]
    cells_satisfy_space = ratio_having_space > 0.09
    if preprocess:
        is_category = _is_category_column_pp(c)
    else:
        is_category = _is_category_column(c)
    return cells_satisfy_space and not is_category


# Generate the PP meta_feature for offline processing
def compute_pp_meta_features(df, proj_name, project, target_column_name):
    try:
        meta_feature_dict = _collect_pp_meta_features(df, target_column_name)
    except Exception as e:
        print("Could not generate pp meta-features for {}".format(proj_name))
        print("Exception: {}".format(e))
        return None

    # Add the file name and the notebook name
    meta_feature_dict["file_name"] = project.file_name
    meta_feature_dict["notebook_name"] = project.notebook_name
    meta_feature_dict["project_name"] = project.project_name
    meta_feature_dict["accuracy"] = project.accuracy
    meta_feature_dict["csv_name"] = project.csv_name

    # Move the notebook name to the second item in the ordered dict
    meta_feature_dict.move_to_end("accuracy", last=False)
    meta_feature_dict.move_to_end("csv_name", last=False)
    meta_feature_dict.move_to_end("notebook_name", last=False)
    meta_feature_dict.move_to_end("project_name", last=False)
    meta_feature_dict.move_to_end("file_name", last=False)

    return meta_feature_dict


# Generate the PP meta_feature for online processing
def generate_pp_meta_features(
    dataframe: pd.DataFrame,
    target_columns: Union[str, list[str]],
):
    try:
        pp_meta_feature_dict = _collect_pp_meta_features(dataframe, target_columns)
    except Exception as e:
        print("Could not generate meta-features.")
        print("Exception: {}".format(e))
        raise
    interesting_features = search_space.meta_feature_list + [ps_macros.STR_OTHER] + [ps_macros.ALL_MISSING_PRESENCE]
    return {k: v for k, v in pp_meta_feature_dict.items() if k in interesting_features}


# Generate the model meta_feature for online processing
def generate_model_meta_features(
    user_training: pd.DataFrame, target_column_name: Union[str, list[str]], is_clf_task: Literal[0, 1]
) -> MetaFeatures:
    # Generate the meta-features
    try:
        meta_feature_dict = _collect_model_meta_features(user_training, target_column_name)
    except Exception as e:
        print("Could not generate meta-features for")
        print("Exception: {}".format(e))
        raise

    # Write the train data under 'features' directory
    meta_features_df = pd.DataFrame([meta_feature_dict])
    meta_features_df = test_feature_preprocess(meta_features_df, is_clf_task)
    return {k: v[0] for k, v in meta_features_df.to_dict().items()}


def generate_column_meta_features(df_column: pd.DataFrame):
    return _collect_column_meta_features(df_column)
