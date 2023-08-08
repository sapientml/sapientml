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
import re

TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
PRED_DEFAULT = "default"  # predict()
PRED_PROBABILITY = "probability"  # predict_proba() for classifications


class Metric(enum.Enum):
    F1 = "f1"
    R2 = "r2"
    RMSLE = "RMSLE"
    RMSE = "RMSE"
    AUC = "auc"
    Accuracy = "accuracy"
    MAE = "MAE"
    Gini = "Gini"
    LogLoss = "LogLoss"
    ROC_AUC = "ROC_AUC"
    MCC = "MCC"  # Matthews correlation coefficient
    MAP_K = "MAP_"
    QWK = "QWK"  # Quadratic weighted kappa

    @staticmethod
    def get(string) -> str:
        ret = [x.value for x in Metric if string.lower() == x.name.lower()]
        if len(ret) == 1:
            return ret[0]
        elif re.match(rf"{Metric.MAP_K.value}[0-9]+", string):
            return string
        raise NotImplementedError(f"metric name '{string}' is not defined. Available: {str([x.name for x in Metric])}")

    @staticmethod
    def get_default_value(task_type: str) -> str:
        # NOTE: See pipeline_template.Pipeline.create_evaluation_code defines the default value of adaptation_metric.
        if task_type == TASK_REGRESSION:
            return Metric.R2.value
        # if task_type == TASK_CLASSIFICATION:
        return Metric.F1.value

    @staticmethod
    def metric_match_task_type(adaptation_metric: str, task_type: str) -> bool:
        if task_type == "regression":
            if adaptation_metric in metrics_for_classification:
                return False
            elif re.match(rf"{Metric.MAP_K.value}[0-9]+", adaptation_metric):
                return False
        elif task_type == "classification" and adaptation_metric in metrics_for_regression:
            return False
        return True

    @staticmethod
    def metric_support_multioutput(adaptation_metric: str) -> bool:
        if adaptation_metric in metrics_not_support_multioutput:
            return False
        elif re.match(rf"{Metric.MAP_K.value}[0-9]+", adaptation_metric):
            return False
        else:
            return True

    @staticmethod
    def metric_support_multiclass_multioutput(adaptation_metric: str) -> bool:
        if adaptation_metric is Metric.Gini.value:
            return True
        else:
            return False


metric_lower_is_better = [Metric.RMSLE.value, Metric.RMSE.value, Metric.MAE.value, Metric.LogLoss.value]

metric_needing_predict_proba = [
    Metric.LogLoss.value,
    Metric.AUC.value,
    Metric.ROC_AUC.value,
    Metric.MAP_K.value,
    Metric.Gini.value,
]

metrics_for_regression = [Metric.R2.value, Metric.RMSLE.value, Metric.RMSE.value, Metric.MAE.value]

metrics_for_classification = [
    Metric.F1.value,
    Metric.AUC.value,
    Metric.Accuracy.value,
    Metric.Gini.value,
    Metric.LogLoss.value,
    Metric.ROC_AUC.value,
    Metric.MCC.value,
    Metric.MAP_K.value,
    Metric.QWK.value,
]

metrics_not_support_multioutput = [
    Metric.MCC.value,
    Metric.MAP_K.value,
    Metric.QWK.value,
]
