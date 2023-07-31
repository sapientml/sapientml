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

from ...enums import Operator


class Predicate:
    feature_name = ""
    _operator = ""
    _comparison_value = ""

    def __init__(self, feature_name, operator, comparison_value):
        self.feature_name = feature_name
        self._operator = operator
        self._comparison_value = comparison_value

    def evaluate_predicate(self, meta_features):
        try:
            actual_value = meta_features[self.feature_name]
            if actual_value == -1 or actual_value == 0:
                return False
        except Exception:
            return False

        result = False
        if self._operator is Operator.GREATER_THAN:
            result = actual_value > self._comparison_value
        elif self._operator is Operator.GREATER_THAN_OR_EQUAL_TO:
            result = actual_value >= self._comparison_value
        elif self._operator is Operator.EQUAL_TO:
            result = actual_value == self._comparison_value
        elif self._operator is Operator.LESS_THAN:
            result = actual_value < self._comparison_value
        elif self._operator is Operator.LESS_THAN_OR_EQUAL_TO:
            result = actual_value <= self._comparison_value

        return result
