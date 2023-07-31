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

drop_label_list = [
    "PREPROCESS:MissingValues:dropna:pandas",
    "PREPROCESS:MissingValues:notnull:pandas",
    "PREPROCESS:MissingValues:isnull:pandas",
]
filler_label = [
    "PREPROCESS:MissingValues:fillna:pandas",
    "PREPROCESS:MissingValues:SimpleImputer:sklearn",
    "PREPROCESS:MissingValues:KNNImputer:sklearn",
    "PREPROCESS:MissingValues:replace:pandas",
    "PREPROCESS:MissingValues:random:custom",
    "PREPROCESS:MissingValues:interpolate:sklearn",
]
in_place_converter = [
    "PREPROCESS:Category:LabelEncoder:sklearn",
    "PREPROCESS:Category:factorize:pandas",
    "PREPROCESS:Category:replace:pandas",
    "PREPROCESS:Category:map:custom",
    "PREPROCESS:Category:apply:pandas",
    "PREPROCESS:Category:custom:pandas",
]
one_hot = [
    "PREPROCESS:Category:get_dummies:pandas",
    "PREPROCESS:Category:OneHotEncoder:sklearn",
    "PREPROCESS:Category:LabelBinarizer:sklearn",
]

text_vect = ["PREPROCESS:Text:CountVectorizer:sklearn", "PREPROCESS:Text:TfidfVectorizer:sklearn"]

scaling = [
    "PREPROCESS:Scaling:STANDARD:sklearn",
    "PREPROCESS:Scaling:MIN_MAX:custom",
    "PREPROCESS:Scaling:MIN_MAX:sklearn",
    "PREPROCESS:Scaling:STANDARD:custom",
    "PREPROCESS:Scaling:Robust:sklearn",
    "PREPROCESS:Scaling:STANDARD:Pandas",
    "PREPROCESS:Scaling:normalize:sklearn",
    "PREPROCESS:Scaling:normalize:Pandas",
    "PREPROCESS:Scaling:STANDARD:pandas",
]

date = [
    "PREPROCESS:GenerateColumn:date:pandas",
    "PREPROCESS:GenerateColumn:DATE:pandas",
    "PREPROCESS:GenerateColumn:DATE:custom",
]

text_processing = [
    "PREPROCESS:Text:lower:pandas",
    "PREPROCESS:Text:remove_non_alpha:custom",
    "PREPROCESS:Text:tokenize:nltk",
    "PREPROCESS:Text:Lemmtize:nltk",
]

balancing = [
    "PREPROCESS:Balancing:SMOTE:imblearn",
    "PREPROCESS:Balancing:resample:custom",
    "PREPROCESS:Balancing:sample:custom",
]

log_transform = [
    "PREPROCESS:Scaling:log1p:numpy",
    "PREPROCESS:Scaling:power:custom",
    "PREPROCESS:Scaling:log:numpy",
    "PREPROCESS:Scaling:sqrt:numpy",
    "PREPROCESS:Scaling:exp:numpy",
    "PREPROCESS:Scaling:log:custom",
    "PREPROCESS:Scaling:power_transform:sklearn",
]
