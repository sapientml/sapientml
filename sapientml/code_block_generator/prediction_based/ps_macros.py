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


FILL = "PREPROCESS:MissingValues:fillna:pandas"
IN_PLACE_CONVERT = "PREPROCESS:Category:LabelEncoder:sklearn"
ONE_HOT = "PREPROCESS:Category:get_dummies:pandas"
VECT = "PREPROCESS:Text:TfidfVectorizer:sklearn"
MISSING = "PREPROCESS:MissingValues:all"
CATG = "PREPROCESS:Category:all"
SCALING = "PREPROCESS:Scaling:STANDARD:sklearn"
DATE = "PREPROCESS:GenerateColumn:DATE:pandas"
LEMMITIZE = "PREPROCESS:TextProcessing:Processing:custom"
BALANCING = "PREPROCESS:Balancing:SMOTE:imblearn"
LOG = "PREPROCESS:Scaling:log:custom"

# Revised meta-features

CATG_PRESENCE = "feature:str_category_presence"
TEXT_PRESENCE = "feature:str_text_presence"
BINARY_CATG_PRESENCE = "feature:str_category_binary_presence"
SMALL_CATG_PRESENCE = "feature:str_category_small_presence"
LARGE_CATG_PRESENCE = "feature:str_category_large_presence"
DATE_PRESENCE = "feature:str_date_presence"
STR_OTHER = "feature:str_other"

MISSING_PRESENCE = "feature:missing_values_presence"
ALL_MISSING_PRESENCE = "feature:all_missing_values"
DATE_PRESENCE = "feature:str_date_presence"

NORMALIZED_MEAN = "feature:max_normalized_mean"
NORMALIZED_STD_DEV = "feature:max_normalized_stddev"
NORMALIZED_VARIATION_ACROSS_COLUMNS = "feature:normalized_variation_across_columns"
IMBALANCE = "feature:target_imbalance_score"
MAX_SKEW = "feature:max_skewness"


TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
