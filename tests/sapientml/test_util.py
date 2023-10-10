import datetime
import math
from uuid import uuid4

import numpy as np
from sapientml.util.json_util import JSONDecoder, JSONEncoder


def test_misc_json_util():
    _json = {
        "array": [np.int_(1), np.int_(2)],
        "float": np.float_(1.0),
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
    }
    JSONEncoder().encode(_json)

    _jsonstr = """{
        "1": null,
        "2": Infinity,
        "3": -Infinity
    }"""
    JSONDecoder().decode(_jsonstr)
