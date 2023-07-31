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

import os

from jinja2 import Environment, FileSystemLoader
from sapientml.params import Dataset, Pipeline, Task

template_env = Environment(
    loader=FileSystemLoader(f"{os.path.dirname(__file__)}/../../../templates/load"), trim_blocks=True
)
ROW_THRESHOLD_FOR_SAMPLING = 100000


def _render(tpl, *args, **kwargs):
    code = tpl.render(*args, **kwargs)
    return "\n".join([line for line in code.split("\n") if len(line) > 0]) + "\n\n"


def generate_code_data_load(dataset: Dataset, task: Task):
    code_validation = code_test = code_train = code_predict = "# *** GENERATED PIPELINE ***\n\n"

    tpl = template_env.get_template("load_data.py.jinja")
    code_validation += _render(tpl, dataset=dataset, task=task, validation=True)
    code_test += _render(tpl, dataset=dataset, task=task, validation=False)
    tpl = template_env.get_template("load_data_train.py.jinja")
    code_train += _render(tpl, dataset=dataset, task=task, script_type="train")
    tpl = template_env.get_template("load_data_predict.py.jinja")
    code_predict += _render(tpl, dataset=dataset, task=task, script_type="predict")

    tpl = template_env.get_template("split.py.jinja")
    code_validation += _render(tpl, task=task, validation=True)
    code_test += _render(tpl, task=task, validation=False)

    tpl = template_env.get_template("subsample.py.jinja")
    code_validation += _render(tpl, task=task, sample_size=ROW_THRESHOLD_FOR_SAMPLING)

    dataset.training_dataframe = dataset.training_dataframe.drop(dataset.ignore_columns, axis=1, errors="ignore")
    if dataset.validation_dataframe is not None:
        dataset.validation_dataframe = dataset.validation_dataframe.drop(
            dataset.ignore_columns, axis=1, errors="ignore"
        )
    if dataset.test_dataframe is not None:
        dataset.test_dataframe = dataset.test_dataframe.drop(dataset.ignore_columns, axis=1, errors="ignore")

    return Pipeline(
        code_for_validation=code_validation,
        code_for_test=code_test,
        code_for_train=code_train,
        code_for_predict=code_predict,
    )
