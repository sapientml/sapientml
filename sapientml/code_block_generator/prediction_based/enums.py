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

import enum


# various operators in decision path for FE/pre-processing meta-models.
class Operator(enum.Enum):
    EQUAL_TO = enum.auto()
    NOT_EQUAL_TO = enum.auto()
    GREATER_THAN = enum.auto()
    GREATER_THAN_OR_EQUAL_TO = enum.auto()
    LESS_THAN = enum.auto()
    LESS_THAN_OR_EQUAL_TO = enum.auto()
