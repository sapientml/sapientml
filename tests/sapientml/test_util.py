import datetime
import math
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from sapientml.params import _normalize_mixed_datetime_columns
from sapientml.util.json_util import JSONDecoder, JSONEncoder


def test_misc_json_util():
    _json = {
        "array": [np.int32(1), np.int32(2)],
        "float": np.float32(1.0),
        "nan": np.nan,
        "ndarray": np.ndarray(shape=(2, 2), dtype=float, order="F"),
        "uuid": uuid4(),
        "date": datetime.date.today(),
        "timedelta": datetime.timedelta(seconds=1),
        "exception": Exception(),
        "dict": {
            "nan": math.nan,
        },
        "str": "abc",
        "int": 123,
    }
    JSONEncoder().encode(_json)

    with pytest.raises(TypeError):

        class UnserializableClass:
            pass

        _json = {"cls": UnserializableClass()}
        JSONEncoder().encode(_json)

    _jsonstr = """{
        "0": 0,
        "1": "nan",
        "2": "inf",
        "3": "-inf",
        "4": {"a": "nan", "b": "inf"},
        "5": ["nan", "-inf"]
    }"""
    _json = {
        "0": 0,
        "1": np.nan,
        "2": np.inf,
        "3": -np.inf,
        "4": {"a": np.nan, "b": np.inf},
        "5": [np.nan, -np.inf],
    }
    assert JSONDecoder().decode(_jsonstr) == _json


# ---------------------------------------------------------------------------
# Unit tests for _normalize_mixed_datetime_columns (sapientml/params.py)
# ---------------------------------------------------------------------------


def test_misc_normalize_mixed_datetime_columns_converts_mixed_col():
    """Object column containing Timestamp values is coerced to datetime64[ns].
    Invalid string entries become NaT."""
    ts = pd.Timestamp("2021-06-01")
    df = pd.DataFrame({"dt": [ts, ts, " ", ts, " "], "num": [1, 2, 3, 4, 5]})
    result = _normalize_mixed_datetime_columns(df)

    assert pd.api.types.is_datetime64_any_dtype(result["dt"])
    assert result["dt"].iloc[0] == ts
    assert pd.isna(result["dt"].iloc[2])
    assert result["num"].tolist() == [1, 2, 3, 4, 5]


def test_misc_normalize_mixed_datetime_columns_all_null_column_skipped():
    """An object column whose non-null values are empty is left unchanged (continue branch)."""
    df = pd.DataFrame({"empty": pd.array([None, None, None], dtype=object), "val": [1, 2, 3]})
    result = _normalize_mixed_datetime_columns(df)

    assert result["empty"].isna().all()
    assert result["empty"].dtype == object


def test_misc_normalize_mixed_datetime_columns_no_timestamps_unchanged():
    """Object column with no Timestamp values is not converted."""
    df = pd.DataFrame({"text": ["foo", "bar", "baz"]})
    result = _normalize_mixed_datetime_columns(df)

    assert result["text"].dtype == object
    assert result["text"].tolist() == ["foo", "bar", "baz"]


def test_misc_normalize_mixed_datetime_columns_no_object_columns():
    """DataFrame with no object-dtype columns is returned unchanged."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})
    result = _normalize_mixed_datetime_columns(df)

    assert result["a"].tolist() == [1, 2, 3]
    assert result["b"].tolist() == [1.1, 2.2, 3.3]


def test_misc_normalize_mixed_datetime_columns_does_not_mutate_input():
    """The original DataFrame must not be mutated."""
    ts = pd.Timestamp("2021-06-01")
    df = pd.DataFrame({"dt": [ts, " ", ts]})
    original_dtype = df["dt"].dtype
    _normalize_mixed_datetime_columns(df)

    assert df["dt"].dtype == original_dtype
