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
    """metrics used for verifying the results accuracy.

    Attributes
    ----------
    F1 : str
        F1 Score metrics
    R2 : str
        R2 score used to evaluate the performance of a regression-based machine learning model.
    RMSLE : str
        Root Mean Squared Logarithmic Error (RMSLE)
    RMSE : str
        Root Mean Squared Error (RMSE)
    AUC : str
        AUC (Area Under the Curve)
    Accuracy : str
        Accuracy is a metric that generally describes how the model performs across all classes
        It is calculated as the ratio between the number of correct predictions to the total number of predictions.
    MAE : str
        Mean Absolute Error (MAE)
    Gini : str
        The Gini Coefficient is used to evaluate the performance of Binary Classifier Models.
        The value of the Gini Coefficient can be between 0 to 1. The higher the Gini coefficient, the better is the model.
    LogLoss : str
        Log loss, also known as logarithmic loss or cross-entropy loss, is a common evaluation metric for binary classification models.
    ROC_AUC : str
        ROC (receiver operating characteristic curve)
        AUC (Area Under the Curve)
        ROC is a probability curve and AUC represents the degree or measure of separability.
    MCC : str
        The Matthews correlation coefficient (MCC)
    MAP_K : str
        Mean Average Precision at K (MAP@K) is one of the most commonly used evaluation metrics for recommender systems and other ranking related classification tasks.
    QWK : str
        QWK (Quadratic weighted kappa) is useful when dealing with imbalanced datasets or when the cost of misclassification varies across classes.

    """

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
        """get method.

        Parameters
        ----------
        string : str

        Results
        -------
        str
            It returns string.

        """
        ret = [x.value for x in Metric if string.lower() == x.name.lower()]
        if len(ret) == 1:
            return ret[0]
        elif re.match(rf"{Metric.MAP_K.value}[0-9]+", string):
            return string
        raise NotImplementedError(f"metric name '{string}' is not defined. Available: {str([x.name for x in Metric])}")

    @staticmethod
    def get_default_value(task_type: str) -> str:
        """get_default_value method.

        Parameters
        ----------
        task_type : str

        Results
        -------
        str
            It returns Metric.F1.value.

        """
        # NOTE: See pipeline_template.Pipeline.create_evaluation_code defines the default value of adaptation_metric.
        if task_type == TASK_REGRESSION:
            return Metric.R2.value
        # if task_type == TASK_CLASSIFICATION:
        return Metric.F1.value

    @staticmethod
    def metric_match_task_type(adaptation_metric: str, task_type: str) -> bool:
        """metric_match_task_type method.

        Parameters
        ----------
        adaptation_metric : str
        task_type : str

        Results
        -------
        bool
            True and otherwise False.

        """
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
        """metric_support_multioutput method.

        Parameters
        ----------
        adaptation_metric : str

        Results
        -------
        bool
            True and otherwise False.

        """
        if adaptation_metric in metrics_not_support_multioutput:
            return False
        elif re.match(rf"{Metric.MAP_K.value}[0-9]+", adaptation_metric):
            return False
        else:
            return True

    @staticmethod
    def metric_support_multiclass_multioutput(adaptation_metric: str) -> bool:
        """metric_support_multiclass_multioutput method.

        Parameters
        ----------
        adaptation_metric : str

        Results
        -------
        bool
            True and otherwise False.

        """
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
