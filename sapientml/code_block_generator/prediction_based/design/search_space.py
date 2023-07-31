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

from .. import ps_macros
from .pp_component_groups import (
    balancing,
    date,
    drop_label_list,
    filler_label,
    in_place_converter,
    log_transform,
    one_hot,
    scaling,
    text_processing,
    text_vect,
)

target_labels = [
    ps_macros.FILL,
    ps_macros.IN_PLACE_CONVERT,
    ps_macros.ONE_HOT,
    ps_macros.VECT,
    ps_macros.DATE,
    ps_macros.LEMMITIZE,
    ps_macros.BALANCING,
    ps_macros.SCALING,
    ps_macros.LOG,
]

# Manually created semantic labels
# Semantic labels are those that cannot be discriminated by our current list of meta-features


label_mapping = {
    # macros.DROP: drop_label_list,
    ps_macros.FILL: filler_label,
    ps_macros.IN_PLACE_CONVERT: in_place_converter,
    ps_macros.ONE_HOT: one_hot,
    ps_macros.VECT: text_vect,
    ps_macros.MISSING: drop_label_list + filler_label,
    ps_macros.CATG: in_place_converter + one_hot,
    ps_macros.DATE: date,
    ps_macros.LEMMITIZE: text_processing,
    ps_macros.SCALING: scaling,
    ps_macros.BALANCING: balancing,
    ps_macros.LOG: log_transform,
}


project_related_metadata = ["file_name", "project_name", "notebook_name", "csv_name", "accuracy", "target_column_name"]

meta_feature_list = [
    ps_macros.CATG_PRESENCE,
    ps_macros.TEXT_PRESENCE,
    ps_macros.BINARY_CATG_PRESENCE,
    ps_macros.SMALL_CATG_PRESENCE,
    ps_macros.LARGE_CATG_PRESENCE,
    ps_macros.MISSING_PRESENCE,
    ps_macros.NORMALIZED_MEAN,
    ps_macros.NORMALIZED_STD_DEV,
    ps_macros.NORMALIZED_VARIATION_ACROSS_COLUMNS,
    ps_macros.DATE_PRESENCE,
    ps_macros.IMBALANCE,
    ps_macros.MAX_SKEW,
]
