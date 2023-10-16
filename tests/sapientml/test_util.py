import datetime
import math
from uuid import uuid4

import numpy as np
import pytest
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
