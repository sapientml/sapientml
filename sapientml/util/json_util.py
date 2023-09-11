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
from datetime import date, datetime, timedelta
from uuid import UUID

import numpy as np


class JSONEncoder(json.JSONEncoder):
    """Encoding Json"""

    def default(self, obj):
        """default method for JSONEncoder.

        Parameters
        ----------
        obj : Instance of the object

        Results
        ------
        Depends upon the instance type
            if instance is integer, it will return int.
            if instance is float, it will return float.
            if instance is string, it will return string.
            if instance is UUID, it will returns string.

        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return str(obj)
        if isinstance(obj, Exception):
            return repr(obj)
        return super(JSONEncoder, self).default(obj)


class JSONDecoder(json.JSONDecoder):
    """Decoding Json"""

    def default(self, obj):
        """default method for JSONEncoder.

        Parameters
        ----------
        obj : Instance of the object

        Results
        ------
        Depends upon the instance
            if instance is nan, it will return np.nan.
            if instance is inf, it will return np.inf.
            if instance is -inf, it will return -np.inf.

        """
        if obj == "nan":
            return np.nan
        if obj == "inf":
            return np.inf
        if obj == "-inf":
            return -np.inf
        return super(JSONDecoder, self).decode(obj)
