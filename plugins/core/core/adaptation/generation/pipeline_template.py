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

import json
import logging
import os
import textwrap
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
from sapientml import macros
from sapientml.util.json_util import JSONEncoder

from ...params import Pipeline
from ...seeding.predictor import name_to_label_mapping

logger = logging.getLogger("sapientml")
env = Environment(
    loader=FileSystemLoader(f"{os.path.dirname(__file__)}/../../../templates"), trim_blocks=True
)

MODEL_IMPORT_LIBRARY_MAP = {
    "XGBRegressor": "xgboost",
    "XGBClassifier": "xgboost",
    "RandomForestRegressor": "sklearn.ensemble",
    "RandomForestClassifier": "sklearn.ensemble",
    "LogisticRegression": "sklearn.linear_model",
    "LinearRegression": "sklearn.linear_model",
    "LinearSVC": "sklearn.svm",
    "LinearSVR": "sklearn.svm",
    "KNeighborsClassifier": "sklearn.neighbors",
    "GradientBoostingRegressor": "sklearn.ensemble",
    "GradientBoostingClassifier": "sklearn.ensemble",
    "ExtraTreesRegressor": "sklearn.ensemble",
    "ExtraTreesClassifier": "sklearn.ensemble",
    "CatBoostRegressor": "catboost",
    "CatBoostClassifier": "catboost",
    "DecisionTreeRegressor": "sklearn.tree",
    "DecisionTreeClassifier": "sklearn.tree",
    "SGDRegressor": "sklearn.linear_model",
    "SGDClassifier": "sklearn.linear_model",
    "LGBMRegressor": "lightgbm",
    "LGBMClassifier": "lightgbm",
    "MLPClassifier": "sklearn.neural_network",
    "MLPRegressor": "sklearn.neural_network",
    "MultinomialNB": "sklearn.naive_bayes",
    "BernoulliNB": "sklearn.naive_bayes",
    "SVC": "sklearn.svm",
    "SVR": "sklearn.svm",
    "AdaBoostClassifier": "sklearn.ensemble",
    "AdaBoostRegressor": "sklearn.ensemble",
    "Lasso": "sklearn.linear_model",
    "GaussianNB": "sklearn.naive_bayes",
}

NO_RANDOM_SEED_MODELS = [
    "SVR",
    "LinearRegression",
    "MultinomialNB",
    "GaussianNB",
    "BernoulliNB",
]

NO_TUNABLE_PARAMS_MODELS = [
    "LinearRegression",
    "MultinomialNB",
    "GaussianNB",
    "BernoulliNB",
]


def load_json(file_name):
    with open(file_name, "r", encoding="utf-8") as input_file:
        content = json.load(input_file)
    return content


def is_allowed_to_apply_to_target(label_name: str) -> bool:
    return "PREPROCESS:Scaling:log" in label_name


class PipelineTemplate(BaseModel):
    pipeline: Pipeline

    def _render(self, tpl, *args, **kwargs):
        code = tpl.render(*args, **kwargs)
        return "\n".join([line for line in code.split("\n") if len(line) > 0])

    def generate(self):
        pipeline = self.pipeline
        if pipeline.model is None:
            return
        model_name = pipeline.model.label_name.split(":")[2]
        is_boolean_target = any(
            [pipeline.dataset_summary.columns[col].dtype == "bool" for col in pipeline.task.target_columns]
        )
        if model_name == "CatBoostClassifier" and is_boolean_target:
            target2string = True
        else:
            target2string = False

        # Use tpl.render but self._render because keep blank lines.
        tpl = env.get_template("other_templates/store_prediction_columns.py.jinja")
        pipeline.pipeline_json["store_prediction_columns"]["code"] = self._render(tpl, pipeline=pipeline)

        if len(pipeline.task.ignore_columns) > 0:
            pipeline.train_ignore_columns = pipeline.task.ignore_columns
            pipeline.test_ignore_columns = pipeline.task.ignore_columns
            tpl = env.get_template("other_templates/drop_columns.py.jinja")
            pipeline.pipeline_json["discard_columns"]["code"] = self._render(tpl, pipeline=pipeline)
            tpl = env.get_template("other_templates/drop_columns_train.py.jinja")
            pipeline.pipeline_json["discard_columns"]["code_train"] = self._render(tpl, pipeline=pipeline)
            tpl = env.get_template("other_templates/drop_columns_predict.py.jinja")
            pipeline.pipeline_json["discard_columns"]["code_predict"] = self._render(tpl, pipeline=pipeline)

        tpl = env.get_template("other_templates/target_separation_validation.py.jinja")
        pipeline.pipeline_json["target_separation"]["code_validation"] = self._render(tpl, pipeline=pipeline)
        tpl = env.get_template("other_templates/target_separation_test.py.jinja")
        pipeline.pipeline_json["target_separation"]["code_test"] = self._render(tpl, pipeline=pipeline)
        tpl = env.get_template("other_templates/target_separation_train.py.jinja")
        pipeline.pipeline_json["target_separation"]["code_train"] = self._render(tpl, pipeline=pipeline)
        tpl = env.get_template("other_templates/target_separation_predict.py.jinja")
        pipeline.pipeline_json["target_separation"]["code_predict"] = self._render(tpl, pipeline=pipeline)

        if pipeline.inverse_target:
            tpl = env.get_template("other_templates/inverse_target.py.jinja")
            pipeline.pipeline_json["inverse_target"]["code"] = self._render(tpl, pipeline=pipeline)

        tpl = env.get_template("other_templates/evaluation.py.jinja")
        code = self._render(tpl, pipeline=pipeline, target2string=target2string, macros=macros)
        pipeline.pipeline_json["evaluation"]["code_validation"] = code
        pipeline.pipeline_json["evaluation"]["code_test"] = textwrap.indent(code, "    ")
        tpl = env.get_template("other_templates/prediction_result.py.jinja")
        pipeline.pipeline_json["output_prediction"]["code"] = self._render(tpl, pipeline=pipeline, macros=macros)

        flag_hyperparameter_tuning = (
            pipeline.config.hyperparameter_tuning and model_name not in NO_TUNABLE_PARAMS_MODELS
        )
        if flag_hyperparameter_tuning:
            tpl = env.get_template("model_templates/hyperparameters.py.jinja")
            pipeline.pipeline_json["hyperparameters"]["code"] = self._render(tpl, model_name=model_name)

            tpl = env.get_template("model_templates/hyperparameters_default_value.py.jinja")
            pipeline.pipeline_json["hyperparameters_default_value"]["code"] = self._render(tpl, model_name=model_name)

            tpl = env.get_template("other_templates/hyperparameter_tuning_evaluation.py.jinja")
            pipeline.pipeline_json["hyperparameter_tuning_evaluation"]["code"] = self._render(
                tpl, pipeline=pipeline, target2string=target2string, macros=macros
            )

            self.populate_hyperparameter_tuning()
        elif pipeline.config.hyperparameter_tuning and model_name in NO_TUNABLE_PARAMS_MODELS:
            logger.debug(
                "'hyperparameter_tuning' is set to True, but the candidate model("
                + model_name
                + ") doesn't have tunable parameters."
            )

        self.populate_model()

        if pipeline.config.permutation_importance:
            tpl = env.get_template("other_templates/permutation_importance.py.jinja")
            pipeline.pipeline_json["permutation_importance"]["code"] = self._render(
                tpl, pipeline=pipeline, target2string=target2string
            )

        tpl_validation = env.get_template("pipeline_validation.py.jinja")
        pipeline.validation = tpl_validation.render(
            pipeline_json=pipeline.pipeline_json,
            pipeline=pipeline,
            flag_hyperparameter_tuning=flag_hyperparameter_tuning,
        )
        tpl_test = env.get_template("pipeline_test.py.jinja")
        pipeline.test = tpl_test.render(
            pipeline_json=pipeline.pipeline_json,
            pipeline=pipeline,
            flag_hyperparameter_tuning=flag_hyperparameter_tuning,
        )

        tpl_train = env.get_template("pipeline_train.py.jinja")
        pipeline.train = tpl_train.render(
            pipeline_json=pipeline.pipeline_json,
            pipeline=pipeline,
            flag_hyperparameter_tuning=flag_hyperparameter_tuning,
        )
        tpl_predict = env.get_template("pipeline_predict.py.jinja")
        pipeline.predict = tpl_predict.render(
            pipeline_json=pipeline.pipeline_json,
            pipeline=pipeline,
            flag_hyperparameter_tuning=flag_hyperparameter_tuning,
        )

    def add_processing_components(self, component, type, training_dataframe, test_dataframe):
        snippets = self.generate_snippet(component, training_dataframe, test_dataframe)
        snippets_train = self.generate_snippet(component, training_dataframe, test_dataframe, "train")
        snippets_predict = self.generate_snippet(component, training_dataframe, test_dataframe, "predict")

        if len(snippets) > 0:
            # prepend explanation to the snippet of the component; first snippet in case of multiple snippets
            component_json, explanation = self.create_preprocessing_component_explanation(component)
            snippets[0] = f"{explanation}\n{snippets[0]}"

            component_json["code"] = snippets.copy()
            component_json["code_train"] = snippets_train.copy()
            component_json["code_predict"] = snippets_predict.copy()
            self.pipeline.pipeline_json[type][component.label_name] = component_json

    def generate_snippet(self, component, training_dataframe, test_dataframe, train_predict_flag=""):
        adapted_snippets = list()

        pipeline = self.pipeline

        relevant_cols = component.relevant_columns
        if not is_allowed_to_apply_to_target(component.label_name):
            relevant_cols = sorted(list(set(relevant_cols) - set(pipeline.task.target_columns)))
        api_label = component.label_name.split(":")[2]

        if len(relevant_cols) == 0 and api_label not in ["STANDARD", "SMOTE"]:
            logger.debug("No relevant columns found for " + api_label)
            return adapted_snippets

        # read component template(s)
        template_files = []
        path = str(
            Path(os.path.dirname(__file__)) / "../../../templates/preprocessing_templates"
        )
        for file in os.listdir(path):
            if file.startswith(api_label):
                if api_label in ["SMOTE"] and train_predict_flag in ["", "train"]:
                    template_files.append(f"preprocessing_templates/{file}")
                elif train_predict_flag == "":
                    if ("train" not in file) and ("predict" not in file):
                        template_files.append(f"preprocessing_templates/{file}")
                elif train_predict_flag != "":
                    if train_predict_flag in file:
                        template_files.append(f"preprocessing_templates/{file}")

        if len(template_files) == 1:
            tpl = env.get_template(template_files[0])
            adapted_snippets.append(
                self._render(
                    tpl,
                    columns=relevant_cols,
                    train_dataset=training_dataframe,
                    test_dataset=test_dataframe,
                    pipeline=pipeline,
                )
            )
        elif len(template_files) > 1:
            # do more analysis to figure out the right template
            if "type" in template_files[0]:
                numeric_cols = []
                other_cols = []

                numeric_dtypes = ["int16", "int32", "int64", "float16", "float32", "float64"]
                for c in relevant_cols:
                    c_dtype = pipeline.all_columns_datatypes[c]
                    if c_dtype in numeric_dtypes:
                        numeric_cols.append(c)
                    else:
                        other_cols.append(c)

                if len(numeric_cols) > 0:
                    template_filename = [x for x in template_files if "numeric" in x]
                    tpl = env.get_template(template_filename[0])
                    cols_almost_missing_numeric = pipeline.dataset_summary.cols_almost_missing_numeric or []
                    adapted_snippets.append(
                        self._render(
                            tpl,
                            cols_almost_missing_numeric=sorted(
                                list(set(numeric_cols) & set(cols_almost_missing_numeric))
                            ),  # because cols_almost_missing_numeric can include columns in ignore_columns, while numeric_cols doesn't include them.
                            columns=sorted(list(set(numeric_cols) - set(cols_almost_missing_numeric))),
                            train_dataset=training_dataframe,
                            test_dataset=test_dataframe,
                        )
                    )
                if len(other_cols) > 0:
                    template_filename = [x for x in template_files if "string" in x]
                    tpl = env.get_template(template_filename[0])
                    cols_almost_missing_string = pipeline.dataset_summary.cols_almost_missing_string or []
                    adapted_snippets.append(
                        self._render(
                            tpl,
                            cols_almost_missing_string=sorted(
                                list(set(other_cols) & set(cols_almost_missing_string))
                            ),  # because cols_almost_missing_string can include columns in ignore_columns, while other_cols doesn't include them.
                            columns=sorted(list(set(other_cols) - set(cols_almost_missing_string))),
                            train_dataset=training_dataframe,
                            test_dataset=test_dataframe,
                        )
                    )
        else:
            logger.debug("ERROR: No template found for component ", api_label)

        return adapted_snippets

    def populate_hyperparameter_tuning(self):
        pipeline = self.pipeline
        if pipeline.model is None:
            return
        model_name = pipeline.model.label_name.split(":")[2]
        # change "predict" to "predict_proba", e.g., for metric = LogLoss, ROC_AUC, Gini since they require probability to be calculated
        if pipeline.adaptation_metric and (
            pipeline.adaptation_metric in macros.metric_needing_predict_proba
            or pipeline.adaptation_metric.startswith(macros.Metric.MAP_K.value)
            or pipeline.config.predict_option == macros.PRED_PROBABILITY
        ):
            tpl = env.get_template("model_templates/classification_post_process.jinja")
            flag_predict_proba = True
            binary_classification_snippet = self._render(tpl, pipeline=pipeline)
        else:
            binary_classification_snippet = ""
            flag_predict_proba = False

        flag_no_random_seed_model = model_name in NO_RANDOM_SEED_MODELS
        _is_multioutput_regression = (
            pipeline.task.task_type == macros.TASK_REGRESSION and len(pipeline.task.target_columns) > 1
        )
        _is_multioutput_classification = (
            pipeline.task.task_type == macros.TASK_CLASSIFICATION and len(pipeline.task.target_columns) > 1
        )

        tpl = env.get_template("model_templates/hyperparameter_tuning.py.jinja")
        snippet = self._render(
            tpl,
            pipeline=pipeline,
            macros=macros,
            import_library=MODEL_IMPORT_LIBRARY_MAP[model_name],
            model_name=model_name,
            params=pipeline.pipeline_json["hyperparameters"]["code"],
            evaluation=pipeline.pipeline_json["hyperparameter_tuning_evaluation"]["code"],
            enqueue_default_hyperparameters=pipeline.pipeline_json["hyperparameters_default_value"]["code"],
            binary_classification_snippet=binary_classification_snippet,
            flag_no_random_seed_model=flag_no_random_seed_model,
            is_multioutput_regression=_is_multioutput_regression,
            is_multioutput_classification=_is_multioutput_classification,
            flag_predict_proba=flag_predict_proba,
            timeout=pipeline.config.hyperparameter_tuning_timeout
            if pipeline.config.hyperparameter_tuning_timeout > 0
            else None,
        )
        pipeline.pipeline_json["hyperparameter_optimization"]["code"] = snippet

    def populate_model(self):
        pipeline = self.pipeline
        if pipeline.model is None:
            return
        model_component_json = dict()
        model_name = pipeline.model.label_name.split(":")[2]
        tpl = env.get_template("model_templates/model.py.jinja")
        tpl_train = env.get_template("model_templates/model_train.py.jinja")
        tpl_predict = env.get_template("model_templates/model_predict.py.jinja")

        flag_no_random_seed_model = model_name in NO_RANDOM_SEED_MODELS
        flag_hyperparameter_tuning = (
            pipeline.config.hyperparameter_tuning and model_name not in NO_TUNABLE_PARAMS_MODELS
        )
        if flag_hyperparameter_tuning and flag_no_random_seed_model:
            model_arg = "HPO_noRandomSeed"
        elif flag_hyperparameter_tuning and not flag_no_random_seed_model:
            model_arg = "HPO_RandomSeed"
        elif not flag_hyperparameter_tuning and flag_no_random_seed_model:
            model_arg = "noHPO_noRandomSeed"
        else:
            model_arg = "noHPO_RandomSeed"
        flag_predict_proba = (
            (pipeline.adaptation_metric in macros.metric_needing_predict_proba)
            or (pipeline.adaptation_metric.startswith(macros.Metric.MAP_K.value))
            or (pipeline.config.predict_option == macros.PRED_PROBABILITY)
        )
        _is_multioutput_regression = (
            pipeline.task.task_type == macros.TASK_REGRESSION and len(pipeline.task.target_columns) > 1
        )
        _is_multioutput_classification = (
            pipeline.task.task_type == macros.TASK_CLASSIFICATION and len(pipeline.task.target_columns) > 1
        )
        snippet = self._render(
            tpl,
            pipeline=pipeline,
            import_library=MODEL_IMPORT_LIBRARY_MAP[model_name],
            model_name=model_name,
            params=pipeline.model.hyperparameters or "",
            model_arg=model_arg,
            flag_predict_proba=flag_predict_proba,
            is_multioutput_regression=_is_multioutput_regression,
            is_multioutput_classification=_is_multioutput_classification,
            metric_needing_predict_proba=macros.metric_needing_predict_proba,
        )
        snippet_train = self._render(
            tpl_train,
            pipeline=pipeline,
            import_library=MODEL_IMPORT_LIBRARY_MAP[model_name],
            model_name=model_name,
            params=pipeline.model.hyperparameters or "",
            model_arg=model_arg,
            flag_predict_proba=flag_predict_proba,
            is_multioutput_regression=_is_multioutput_regression,
            is_multioutput_classification=_is_multioutput_classification,
            metric_needing_predict_proba=macros.metric_needing_predict_proba,
        )
        snippet_predict = self._render(
            tpl_predict,
            pipeline=pipeline,
            import_library=MODEL_IMPORT_LIBRARY_MAP[model_name],
            model_name=model_name,
            params=pipeline.model.hyperparameters or "",
            model_arg=model_arg,
            flag_predict_proba=flag_predict_proba,
            is_multioutput_regression=_is_multioutput_regression,
            is_multioutput_classification=_is_multioutput_classification,
            metric_needing_predict_proba=macros.metric_needing_predict_proba,
        )

        # change "predict" to "predict_proba", e.g., for metric = LogLoss, ROC_AUC, Gini since they require probability to be calculated
        if pipeline.adaptation_metric and (
            pipeline.adaptation_metric in macros.metric_needing_predict_proba
            or pipeline.adaptation_metric.startswith(macros.Metric.MAP_K.value)
            or pipeline.config.predict_option == macros.PRED_PROBABILITY
        ):
            snippet = snippet.replace("predict", "predict_proba")
            tpl = env.get_template("model_templates/classification_post_process.jinja")
            snippet += "\n" + self._render(tpl, pipeline=pipeline)

            snippet_predict = snippet_predict.replace("predict", "predict_proba")
            tpl_predict = env.get_template("model_templates/classification_post_process.jinja")
            snippet_predict += "\n" + self._render(tpl_predict, pipeline=pipeline)

        model_component_json["code"] = snippet
        model_component_json["code_train"] = snippet_train
        model_component_json["code_predict"] = snippet_predict

        with open(Path(os.path.dirname(__file__)) / "../../models/feature_importance.json", "r", encoding="utf-8") as f:
            model_feature_weights = json.load(f)

        for key, task_label in name_to_label_mapping.items():
            if pipeline.model.label_name in task_label.values():
                model_name = key
                break

        pipeline.model.meta_features = (
            model_feature_weights[model_name][:5] if model_name in model_feature_weights else []
        )

        # prepend explanation to the model snippet
        snippet = self.create_model_component_explanation(pipeline.model, model_component_json) + snippet
        pipeline.pipeline_json["model"] = model_component_json
        return snippet

    def create_preprocessing_component_explanation(self, component):
        component_description_dict = load_json(
            str(
                Path(os.path.dirname(__file__))
                / "../../../templates/explainability_templates"
                / "component_description.json"
            )
        )
        entry = component_description_dict[component.label_name]
        api_description = entry["api_description"]
        data_shape = entry["data_shape"]
        hyperparameters = entry["hyperparameters"]
        relevant_metafeatures = [o.feature_name for o in component.predicate_objects]

        # special case for standard scaler, since it is applied to the entire dataset
        if (
            component.label_name == "PREPROCESS:Scaling:STANDARD:sklearn"
            or component.label_name == "PREPROCESS:Balancing:SMOTE:imblearn"
        ):
            component.relevant_columns = "all columns in the dataset"

        ctx = dict()
        ctx["target_component_name"] = component.label_name
        ctx["relevant_meta_feature_list"] = relevant_metafeatures
        ctx["relevant_column_list"] = component.relevant_columns
        ctx["api_description"] = api_description
        ctx["data_shape"] = data_shape

        if len(component.components_before) > 0:
            ctx["before_or_after"] = "after"
            ctx["dependent_component_list"] = component.components_before
        elif len(component.components_after) > 0:
            ctx["before_or_after"] = "before"
            ctx["dependent_component_list"] = component.components_after

        if len(component.alternative_components) > 0:
            ctx["alternative_component_list"] = component.alternative_components
            alt_comp_predicates = list()
            comp_predicates = [o.feature_name for o in component.predicate_objects]
            for alt_comp in component.alternative_components:
                alt_comp_predicates.extend([o.feature_name for o in alt_comp.predicate_objects])

            metafeature_1 = ""
            metafeature_2 = ""
            for p, alt_p in zip(comp_predicates, alt_comp_predicates):
                if p == alt_p:
                    continue
                else:
                    metafeature_1 = p
                    metafeature_2 = alt_p
                    break
            ctx["relevant_meta_feature_1"] = metafeature_1
            ctx["relevant_meta_feature_2"] = metafeature_2

        hp_text = ""
        if len(hyperparameters) > 0:
            for hp in hyperparameters:
                hp_text += '\n#\t\t "' + hp["hp_name"] + ": " + hp["hp_values"] + '" :: ' + hp["hp_description"]
        else:
            hp_text = "None"
        ctx["hyperparameters_description"] = hp_text

        tpl = env.get_template("explainability_templates/preprocessing_explanation.py.jinja")
        explanation = self._render(tpl, **ctx)

        # normalize component names
        for old_name, map in component_description_dict.items():
            explanation = explanation.replace(old_name, map["normalized_name"])

        # populate json
        component_json = dict()
        component_json["explanation"] = dict()
        component_json["explanation"]["target_component_name"] = component.label_name
        component_json["explanation"]["relevant_meta_feature_list"] = relevant_metafeatures
        component_json["explanation"]["relevant_column_list"] = component.relevant_columns
        component_json["explanation"]["api_description"] = api_description
        component_json["explanation"]["data_shape"] = data_shape
        component_json["explanation"]["hyperparameters_description"] = hyperparameters
        component_json["explanation"]["alternative_component_list"] = list()
        for alt_comp in component.alternative_components:
            component_json["explanation"]["alternative_component_list"].append(
                {
                    "component_name": alt_comp.label_name,
                    "relevant_meta_feature_list": [o.feature_name for o in alt_comp.predicate_objects],
                }
            )
        component_json["explanation"]["before_dependent_component_list"] = component.components_before
        component_json["explanation"]["after_dependent_component_list"] = component.components_after

        return component_json, explanation

    def create_model_component_explanation(self, component, model_component_json):
        tpl = env.get_template("explainability_templates/model_explanation.py.jinja")
        explanation = self._render(
            tpl, target_component_name=component.label_name, relevant_meta_feature_list=component.meta_features
        )

        # populate json
        model_component_json["explanation"] = dict()
        model_component_json["explanation"]["target_component_name"] = component.label_name
        model_component_json["explanation"]["relevant_meta_feature_list"] = component.meta_features

        return explanation

    def save_pipeline_json(self):
        # json_string = json.dumps(self.pipeline_json)
        # with open(self.output_dir_path + 'pipeline_json.json', 'w') as outfile:
        #     json.dump(json_string, outfile)

        with open(self.pipeline.output_dir_path + "pipeline_json.json", "w", encoding="utf-8") as f:
            json.dump(self.pipeline.pipeline_json, f, cls=JSONEncoder, indent=4)
