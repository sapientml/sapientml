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

import copy
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

from sapientml import macros
from sapientml.params import Config, Task

from ...params import DatasetSummary, ModelLabel, Pipeline, PipelineSkeleton
from .pipeline_template import PipelineTemplate, is_allowed_to_apply_to_target
from .preprocessing_label import PreprocessingLabel

logger = logging.getLogger("sapientml")

preprocessing_threshold = 0.5


class Adaptation:
    pipeline: PipelineTemplate

    PREPROCESSING_AFTER_TARGET_SEPARATION = {
        "PREPROCESS:Category:get_dummies:pandas",
        "PREPROCESS:Scaling:StandardScaler:sklearn",
        "PREPROCESS:Text:CountVectorizer:sklearn",
        "PREPROCESS:Text:TfidfVectorizer:sklearn",
        "PREPROCESS:Scaling:STANDARD:sklearn",
        "PREPROCESS:Scaling:StandardScaler:sklearn",
    }

    PREPROCESSING_AFTER_TRAIN_TEST_SPLIT = {"PREPROCESS:Balancing:SMOTE:imblearn"}

    PREPROCESSING_ONLY_TARGET = {"PREPROCESS:Scaling:log:custom", "PREPROCESS:Balancing:SMOTE:imblearn"}

    def __init__(
        self,
        labels: PipelineSkeleton,
        task: Task,
        dataset_summary: DatasetSummary,
        config: Config,
    ):
        self.labels = labels
        self.task = task
        self.dataset_summary = dataset_summary
        self.config = config
        self.adaptation_metric: Optional[str] = self._get_adaptation_metric_label()

        # Load all the Offline Data
        with open(Path(os.path.dirname(__file__)) / "../artifacts/label_order.json", "r", encoding="utf-8") as f:
            label_order = json.load(f)

        self.label_order = label_order
        self.dataset_summary = dataset_summary
        self.pipeline = PipelineTemplate(
            pipeline=Pipeline(
                task=self.task,
                dataset_summary=self.dataset_summary,
                config=self.config,
            )
        )

    def _get_labels_from_skeleton_predictor(self):
        # fetch preprocessing and model labels
        all_preprocessing_labels = dict()
        model_labels = dict()
        for label_name, details in self.labels.items():
            if label_name.startswith("PREPROCESS"):
                all_preprocessing_labels[label_name] = details
            else:
                model_labels[label_name] = details

        # filter labels based on config setting
        preprocessing_labels = dict(
            filter(
                lambda elem: elem[1]["probability"] > preprocessing_threshold,
                all_preprocessing_labels.items(),
            )
        )

        n_models = self.task.n_models
        if n_models < 1:
            raise ValueError("Please set 'n_models' to a number greater than or equal to 1.")
        model_labels = dict(list(model_labels.items())[0:n_models])

        # resolve conflicting preprocessing labels by keeping the one with highest probability
        conflicting_labels = dict()
        preprocessing_labels, conflicting_labels = self._resolve_conflicting_labels(preprocessing_labels)

        # order preprocessing labels and model labels
        preprocessing_labels, model_labels = self._order_labels(preprocessing_labels, model_labels)

        # convert to preprocessing and model objects
        preprocessing_label_objects: list[PreprocessingLabel] = list()
        model_label_objects: list[ModelLabel] = list()
        for name, details in preprocessing_labels.items():
            pObj = PreprocessingLabel(name, details["features"], details["predicates"])
            labels_before, labels_after = self._get_preprocessing_labels_before_and_after(name, preprocessing_labels)
            pObj.components_before = labels_before
            pObj.components_after = labels_after
            if name in conflicting_labels:
                for label in conflicting_labels[name]:
                    pObj.alternative_components.append(
                        PreprocessingLabel(
                            label,
                            all_preprocessing_labels[label]["features"],
                            all_preprocessing_labels[label]["predicates"],
                        )
                    )

            preprocessing_label_objects.append(pObj)

        for name, details in model_labels.items():
            mObj = ModelLabel(label_name=name)
            model_label_objects.append(mObj)

        return preprocessing_label_objects, model_label_objects

    def _get_preprocessing_labels_before_and_after(self, label, preprocessing_labels):
        labels_before = list()
        labels_after = list()
        for combination in self.label_order:
            arr = combination.split("#")
            first = arr[0]
            second = arr[1]
            if label == second and first in preprocessing_labels:
                labels_before.append(first)
            elif label == first and second in preprocessing_labels:
                labels_after.append(second)
        return labels_before, labels_after

    def _resolve_conflicting_labels(self, preprocessing_labels):
        second_level_labels_dict = defaultdict(list)
        for label in preprocessing_labels:
            second_level = label.split(":")[1]
            second_level_labels_dict[second_level].append(label)
        clean_labels = dict()
        conflicting_labels = dict()
        for second_level in second_level_labels_dict:
            sorted_list = sorted(
                second_level_labels_dict[second_level],
                key=lambda i: preprocessing_labels[i]["probability"],
                reverse=True,
            )
            label_to_keep = sorted_list[0]
            clean_labels[label_to_keep] = preprocessing_labels[label_to_keep]

            if len(sorted_list) > 1:
                conflicting_labels[sorted_list[0]] = sorted_list[1:]

        preprocessing_labels = clean_labels
        return preprocessing_labels, conflicting_labels

    def _order_labels(self, preprocessing_labels, model_labels):
        # sort preprocessing components
        sorted_label_names = self._sort(list(preprocessing_labels.keys()), self.label_order)
        sorted_preprocessing_labels = dict()
        for label in sorted_label_names:
            sorted_preprocessing_labels[label] = preprocessing_labels[label]

        # sort model components
        sorted_model_labels = dict(sorted(model_labels.items(), key=lambda item: item[1], reverse=True))

        return sorted_preprocessing_labels, sorted_model_labels

    def _sort(self, preprocessing_set, label_order):
        n = len(preprocessing_set)

        # Traverse through all array elements
        for i in range(n - 1):
            # range(n) also work but outer loop will repeat one time more than needed.

            # Last i elements are already in place
            for j in range(0, n - i - 1):
                # traverse the array from 0 to n-i-1
                # Swap if the element found is greater
                # than the next element
                combination = preprocessing_set[j + 1] + "#" + preprocessing_set[j]

                if combination in label_order:
                    # logger.debug('combination', combination)
                    preprocessing_set[j], preprocessing_set[j + 1] = preprocessing_set[j + 1], preprocessing_set[j]
        return preprocessing_set

    def _get_adaptation_metric_label(self) -> Optional[str]:
        if self.task.adaptation_metric:
            metric = self.task.adaptation_metric.lower()
            if metric == "f1":
                return macros.Metric.F1.value
            elif metric == "r2":
                return macros.Metric.R2.value
            elif metric == "rmsle":
                return macros.Metric.RMSLE.value
            elif metric == "rmse":
                return macros.Metric.RMSE.value
            elif metric == "auc":
                return macros.Metric.AUC.value
            elif metric == "accuracy":
                return macros.Metric.Accuracy.value
            elif metric == "mae":
                return macros.Metric.MAE.value
            elif metric == "gini":
                return macros.Metric.Gini.value
            elif metric == "logloss":
                return macros.Metric.LogLoss.value
            elif metric == "roc_auc":
                return macros.Metric.ROC_AUC.value
            elif metric == "mcc":
                return macros.Metric.MCC.value
            elif metric == "qwk":
                return macros.Metric.QWK.value
            elif metric.startswith("map_"):
                k = metric.split("_")[1]
                try:
                    if int(k) < 1:
                        raise ValueError("Please set 'MAP_K' to a number greater than or equal to 1.")
                except Exception:
                    raise ValueError("Please set number for K in MAP_K.")
                return macros.Metric.MAP_K.value + k
            else:
                raise ValueError(f"Invalid metric: {metric}")
        else:
            return None

    def _setup_pipeline_basics(self):
        pipeline = self.pipeline.pipeline
        pipeline.adaptation_metric = self.adaptation_metric
        pipeline.is_multi_class_multi_targets = (
            len(self.task.target_columns) > 1 and self.dataset_summary.has_multi_class_targets
        )

        # populate all columns and their datatypes
        all_columns_datatypes = dict()
        for column_name, column in self.dataset_summary.columns.items():
            all_columns_datatypes[column_name] = column.dtype
        pipeline.all_columns_datatypes = all_columns_datatypes
        pipeline.train_column_names = list(self.dataset_summary.columns.keys())
        pipeline.test_column_names = list(self.dataset_summary.columns.keys())

    def _populate_preprocessing_components_in_pipeline(self, preprocessing_components):
        if self.pipeline is None:
            return

        columns = self.dataset_summary.columns

        for component in preprocessing_components:
            # handle special case for SMOTE, don't apply if target columns are > 1. SMOTE fails in such cases
            if "PREPROCESS:Balancing:SMOTE:imblearn" == component.label_name and len(self.task.target_columns) > 1:
                continue

            rel_cols = component.get_relevant_columns(
                self.dataset_summary, self.task.target_columns, self.task.ignore_columns
            )

            if (
                hasattr(self.config, "impute_all")
                and self.config.impute_all is True
                and "PREPROCESS:MissingValues:fillna" in component.label_name
            ):
                bool_cols = [column_name for column_name, column in columns.items() if "bool" in str(column.dtype)]
                datetime_cols = [
                    column_name for column_name, column in columns.items() if "datetime" in str(column.dtype)
                ]
                # remove boolean and datetime columns, since Simple Imputer cannot handle boolean and datetime dtype
                rel_cols = sorted(list(set(rel_cols) - set(bool_cols) - set(datetime_cols)))
                rel_cols = sorted(list(set(rel_cols) - set(self.task.ignore_columns) - set(self.task.target_columns)))

            # handle special case for log transformation if target feature is in relevant columns
            if (
                is_allowed_to_apply_to_target(component.label_name)
                and len(list(set(rel_cols) & set(self.task.target_columns))) > 0
            ):
                if self.task.task_type == macros.TASK_REGRESSION:
                    # add code for inversing log transformation, if scaling on target feature
                    self.pipeline.pipeline.inverse_target = True
                else:
                    # in a classification task, an error is caused when the target is made continuous with log transformation
                    # hence, remove target from relevant columns in this scenario
                    rel_cols = sorted(list(set(rel_cols) - set(self.task.target_columns)))

            # construct component snippet
            if component.label_name in self.PREPROCESSING_AFTER_TARGET_SEPARATION:
                # remove target feature
                rel_cols = sorted(list(set(rel_cols) - set(self.task.target_columns)))

            # store relevant columns in the component after doing some additional processing, if needed
            component.relevant_columns = rel_cols

            # add component to pipeline
            if component.label_name in self.PREPROCESSING_AFTER_TARGET_SEPARATION:
                self.pipeline.add_processing_components(
                    component,
                    type="preprocessing_after_target_separation",
                    training_dataframe="__feature_train",
                    test_dataframe="__feature_test",
                )
            elif component.label_name in self.PREPROCESSING_AFTER_TRAIN_TEST_SPLIT:
                self.pipeline.add_processing_components(
                    component,
                    type="preprocessing_after_train_test_split",
                    training_dataframe="__feature_train",
                    test_dataframe="__feature_test",
                )
            else:
                self.pipeline.add_processing_components(
                    component,
                    type="preprocessing_before_target_separation",
                    training_dataframe="__train_dataset",
                    test_dataframe="__test_dataset",
                )

            # The information is used, for example when Scaler is used and Tfidf is not used.
            # We'd like to change numpy.ndarray of Scaler's output into pandas.DataFrame.
            if "PREPROCESS:Text:TfidfVectorizer" in component.label_name:
                self.pipeline.pipeline.sparse_matrix = True

            id = 1
            for component in (
                list(self.pipeline.pipeline.pipeline_json["preprocessing_before_target_separation"].values())
                + list(self.pipeline.pipeline.pipeline_json["preprocessing_after_target_separation"].values())
                + list(self.pipeline.pipeline.pipeline_json["preprocessing_after_train_test_split"].values())
            ):
                component["id"] = id
                id += len(component["code"])

    def _populate_model_components_in_pipeline(self, models: list[ModelLabel]) -> list[PipelineTemplate]:
        pipeline_list: list[PipelineTemplate] = []
        for model_obj in models:
            if self.pipeline.pipeline.adaptation_metric is not None and (
                self.pipeline.pipeline.adaptation_metric in macros.metric_needing_predict_proba
                or self.pipeline.pipeline.adaptation_metric.startswith(macros.Metric.MAP_K.value)
                or self.config.predict_option == macros.PRED_PROBABILITY
            ):
                model_obj.predict_proba = True

            # first variant with default hyperparameters
            pipeline_variant = copy.deepcopy(self.pipeline)
            pipeline_variant.pipeline.model = model_obj
            pipeline_list.append(pipeline_variant)

        return pipeline_list

    def run_adaptation(self) -> list[Pipeline]:
        # SapientML default run
        self._setup_pipeline_basics()
        preprocessing, model = self._get_labels_from_skeleton_predictor()
        self._populate_preprocessing_components_in_pipeline(preprocessing)
        pipeline_list = self._populate_model_components_in_pipeline(model)
        for pipeline in pipeline_list:
            pipeline.generate()

        return [pipeline.pipeline for pipeline in pipeline_list]
