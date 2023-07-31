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

import collections
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sapientml.params import Task

from ..design import search_space
from ..params import DatasetSummary, PipelineSkeleton

name_to_label_mapping = {
    "random forest": {
        "c": "MODEL:Classifier:RandomForestClassifier:sklearn",
        "r": "MODEL:Regressor:RandomForestRegressor:sklearn",
    },
    "extra tree": {
        "c": "MODEL:Classifier:ExtraTreesClassifier:sklearn",
        "r": "MODEL:Regressor:ExtraTreesRegressor:sklearn",
    },
    "lightgbm": {"c": "MODEL:Classifier:LGBMClassifier:lightgbm", "r": "MODEL:Regressor:LGBMRegressor:lightgbm"},
    "xgboost": {"c": "MODEL:Classifier:XGBClassifier:xgboost", "r": "MODEL:Regressor:XGBRegressor:xgboost"},
    "catboost": {
        "c": "MODEL:Classifier:CatBoostClassifier:catboost",
        "r": "MODEL:Regressor:CatBoostRegressor:catboost",
    },
    "gradient boosting": {
        "c": "MODEL:Classifier:GradientBoostingClassifier:sklearn",
        "r": "MODEL:Regressor:GradientBoostingRegressor:sklearn",
    },
    "adaboost": {"c": "MODEL:Classifier:AdaBoostClassifier:sklearn", "r": "MODEL:Regressor:AdaBoostRegressor:sklearn"},
    "decision tree": {
        "c": "MODEL:Classifier:DecisionTreeClassifier:sklearn",
        "r": "MODEL:Regressor:DecisionTreeRegressor:sklearn",
    },
    "svm": {"c": "MODEL:Classifier:SVC:sklearn", "r": "MODEL:Regressor:SVR:sklearn"},
    "linear svm": {"c": "MODEL:Classifier:LinearSVC:sklearn", "r": "MODEL:Regressor:LinearSVR:sklearn"},
    "logistic/linear regression": {
        "c": "MODEL:Classifier:LogisticRegression:sklearn",
        "r": "MODEL:Regressor:LinearRegression:sklearn",
    },
    "lasso": {"r": "MODEL:Regressor:Lasso:sklearn"},
    "sgd": {"c": "MODEL:Classifier:SGDClassifier:sklearn", "r": "MODEL:Regressor:SGDRegressor:sklearn"},
    "mlp": {"c": "MODEL:Classifier:MLPClassifier:sklearn", "r": "MODEL:Regressor:MLPRegressor:sklearn"},
    "multinomial nb": {"c": "MODEL:Classifier:MultinomialNB:sklearn"},
    "gaussian nb": {"c": "MODEL:Classifier:GaussianNB:sklearn"},
    "bernoulli nb": {"c": "MODEL:Classifier:BernoulliNB:sklearn"},
}


class Predicate:
    def __init__(self, feature_name, value, threshold, op):
        self.feature_name = feature_name
        self.value = value
        self.threshold = threshold
        self.op = op

    def __repr__(self):
        return self.feature_name + "(" + self.value + ")" + " " + self.op + " " + self.threshold

    def __str__(self):
        return self.feature_name + "(" + self.value + ")" + " " + self.op + " " + self.threshold


def get_decision_path(clf, X):
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_indicator = clf.decision_path(X)
    leaf_id = clf.apply(X)

    rules = []

    for row_index in range(X.shape[0]):
        sample_id = row_index
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]]

        predicates = []
        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                continue

            # check if value of the split feature for sample 0 is below threshold
            if X.iloc[sample_id, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            predicates.append(
                Predicate(
                    X.columns[feature[node_id]], X.iloc[sample_id, feature[node_id]], threshold[node_id], threshold_sign
                )
            )
        rules.append(predicates)
    return rules


def _predict_preprocessors(pp_models, meta_features: pd.DataFrame, target_labels: list[str]) -> PipelineSkeleton:
    for col in meta_features.columns:
        meta_features[col].fillna(0, inplace=True)
    output = meta_features[search_space.meta_feature_list].copy()
    labels: list[str] = []
    rules = {}
    for label, model_info in pp_models.items():
        if label not in target_labels:
            continue
        labels.append(label)

        all_feat_cols = model_info["relevant_features"]

        final_meta_features = meta_features[all_feat_cols]

        prediction = model_info["model"].predict(final_meta_features)

        prediction_proba = model_info["model"].predict_proba(final_meta_features)[:, 1]
        rules[label] = get_decision_path(model_info["model"], final_meta_features)
        output[label] = prediction
        output[label + "_proba"] = prediction_proba

    pp_labels = collections.OrderedDict()
    for index, row in output.iterrows():
        labels_info = {}
        for label in labels:
            predicates_obj = rules[label][index]
            predicates = []
            for predicate_obj in predicates_obj:
                predicates.append(
                    {
                        "feature_name": predicate_obj.feature_name,
                        "operator": predicate_obj.op,
                        "threshold": predicate_obj.threshold,
                        "actual_value": predicate_obj.value,
                    }
                )
            label_info = {
                "probability": row[label + "_proba"],
                "features": pp_models[label]["relevant_features"],
                "predicates": predicates,
            }
            labels_info[label] = label_info
        pp_labels = labels_info

    return pp_labels


def _predict_models(
    m_model, task_type: str, m_meta_features_test: pd.DataFrame, preprocessor_labels: PipelineSkeleton
) -> PipelineSkeleton:
    meta_features_test = m_meta_features_test[[x for x in m_meta_features_test.columns if x.startswith("feature:")]]
    meta_features = meta_features_test.fillna(0)

    predict_proba = m_model[0].predict_proba(meta_features) + m_model[1].predict_proba(meta_features)

    index_ranking = predict_proba.argsort(axis=1)[:, ::-1]
    model_ranking = np.array(m_model[0].classes_)[index_ranking]

    model_ranking = [
        [
            name_to_label_mapping[item][task_type]
            if item in name_to_label_mapping and task_type in name_to_label_mapping[item]
            else None
            for item in preds
        ]
        for preds, task_type in zip(model_ranking, task_type)
    ]
    columns = ["Pred " + str(i + 1) for i in range(len(model_ranking[0]))]
    predictions_df = pd.DataFrame(model_ranking, columns=columns)

    import copy

    all_labels = copy.deepcopy(preprocessor_labels)
    for index, row in predictions_df.iterrows():
        labels = collections.OrderedDict()
        for i, column in enumerate(columns):
            labels[row[column]] = predict_proba[index][index_ranking[index][i]]
        if "ILLEGAL PREDICTION" in labels:
            del labels["ILLEGAL PREDICTION"]

        if None in labels:
            del labels[None]

        all_labels.update(labels)
    return all_labels


def dict_to_df(d) -> pd.DataFrame:
    return pd.DataFrame.from_dict({0: d}, orient="index", columns=d.keys())


def predict(task: Task, dataset_summary: DatasetSummary) -> PipelineSkeleton:
    task_type = task.task_type
    p_meta_feature_test = dict_to_df(dataset_summary.meta_features_pp)
    m_meta_feature_test = dict_to_df(dataset_summary.meta_features_m)
    with open(Path(os.path.dirname(__file__)) / "../models/pp_models.pkl", "rb") as f:
        pp_model = pickle.load(f)

    with open(Path(os.path.dirname(__file__)) / "../models/mp_model_1.pkl", "rb") as f1:
        with open(Path(os.path.dirname(__file__)) / "../models/mp_model_2.pkl", "rb") as f2:
            m_model = (pickle.load(f1), pickle.load(f2))

    preprocessor_labels = _predict_preprocessors(pp_model, p_meta_feature_test, search_space.target_labels)
    all_labels = _predict_models(m_model, task_type, m_meta_feature_test, preprocessor_labels)

    return all_labels
