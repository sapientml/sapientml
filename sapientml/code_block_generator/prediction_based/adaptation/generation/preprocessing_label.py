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
from .predicate import Predicate


class PreprocessingLabel:
    def __init__(self, label_name, meta_features, predicates):
        self.label_name = label_name
        self.meta_features = meta_features
        self.predicate_objects = list()
        self._build_predicate_objects(predicates)
        self.relevant_columns = list()
        self.components_before = list()
        self.components_after = list()
        self.alternative_components = list()

    def __str__(self):
        return self.label_name

    def __repr__(self):
        return str(self)

    def _build_predicate_objects(self, predicates):
        for pred in predicates:
            feature_name = pred["feature_name"]
            operator = self._get_operator(pred["operator"])
            comparison_value = pred["threshold"]
            p = Predicate(feature_name, operator, comparison_value)
            self.predicate_objects.append(p)

    def _get_operator(self, op_string):
        if op_string == ">":
            return Operator.GREATER_THAN
        elif op_string == ">=":
            return Operator.GREATER_THAN_OR_EQUAL_TO
        elif op_string == "<":
            return Operator.LESS_THAN
        elif op_string == "<=":
            return Operator.LESS_THAN_OR_EQUAL_TO
        elif op_string == "==" or op_string == "=":
            return Operator.EQUAL_TO
        else:
            return Operator.NOT_EQUAL_TO

    def get_relevant_columns(self, dataset_summary, target, ignore_columns):
        rel_columns_list = []

        # approach 1: conjunction: a column is relavant if and only if all of the predicates applicable to that component are true
        # approach 2: disjunction: a column is relavant if and only if at least one of the predicates applicable to that component are true
        approach = 2

        for column_name, column in dataset_summary.columns.items():
            if column_name in ignore_columns:
                continue

            # error handling for log transform: don't apply if any col value <= 0
            if "PREPROCESS:Scaling:log" in self.label_name:
                if column.has_negative_value:
                    continue

            result = list()  # holds boolean results of all predicates applicable to a column
            for p in self.predicate_objects:
                # special handling of "target_imbalance_score" feature, since it should only be applied on target column
                if p.feature_name == "feature:target_imbalance_score":
                    if column_name not in target:
                        result.append(False)
                        continue
                result.append(p.evaluate_predicate(column.meta_features))

            if approach == 1:  # conjunction
                if all(result):
                    rel_columns_list.append(column_name)
            elif approach == 2:  # disjunction
                if any(result):
                    rel_columns_list.append(column_name)

            # # approach 1: a column is relavant if and only if all of the predicates applicable to that component are true
            # if approach == 1:
            #     result1 = True
            #     for p in self.predicate_objects:
            #         result1 = result1 and p.evaluate_predicate(df[[column_name]])
            #         if not result1:
            #             break
            #     if result1:
            #         rel_columns_list.append(column_name)
            #
            # # approach 2: a column is relavant if and only if at least one of the predicates applicable to that component are true (current)
            # elif approach == 2:
            #     result2 = False
            #     for p in self.predicate_objects:
            #         result2 = result2 or p.evaluate_predicate(df[[column_name]])
            #         if result2:
            #             rel_columns_list.append(column_name)
            #             break
        return rel_columns_list
