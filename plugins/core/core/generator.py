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

from pathlib import Path
from importlib.metadata import entry_points
from typing import Tuple
from sapientml.generator import PipelineGenerator, CodeBlockGenerator
from sapientml.params import Config, Task, Dataset, Code
from sapientml.util.logging import setup_logger

from . import ps_macros
from .adaptation.generation.template_based_adaptation import Adaptation
from .params import Pipeline, summarize_dataset
from .seeding.predictor import predict

model_dir_path_default = Path(__file__).parent / "models"

logger = setup_logger()


class SapientMLGenerator(PipelineGenerator, CodeBlockGenerator):
    def __init__(self, config: Config):
        CodeBlockGenerator.__init__(self, config)
        eps = entry_points(group="code_block_generator")
        self.loaddata = eps['loaddata'].load()(config)
        self.preprocess = eps['preprocess'].load()(config)

    def generate_pipeline(self, dataset: Dataset, task: Task) -> list[Code]:
        dataset, loaddata_block = self.loaddata.generate_code(dataset, task)
        dataset, preprocess_block = self.preprocess.generate_code(dataset, task)
        code_block = loaddata_block + preprocess_block
        dataset, sapientml_results = self.generate_code(dataset, task)

        result_pipelines: list[Code] = []
        for pipeline in sapientml_results:
            pipeline.validation = code_block.validation + pipeline.validation
            pipeline.test = code_block.test + pipeline.test
            pipeline.train = code_block.train + pipeline.train
            pipeline.predict = code_block.predict + pipeline.predict
            result_pipelines.append(pipeline)
        return result_pipelines

    def generate_code(self, dataset: Dataset, task: Task) -> Tuple[Dataset, list[Pipeline]]:
        df = dataset.training_dataframe
        dataset_summary = summarize_dataset(df, task)  # type: ignore
        if dataset_summary.has_inf_value_targets:
            raise ValueError("Stopped generation because target columns have infinity value.")

        # discard columns with analysis
        # NOTE: The following code modify task.ignore_columns because ignore_columns is the same instance as task.ignore_columns.
        # 1. columns marked as STR_OTHER
        if ps_macros.STR_OTHER in dataset_summary.meta_features_pp:
            undetermined_column_names = dataset_summary.meta_features_pp[ps_macros.STR_OTHER]
            if isinstance(undetermined_column_names, list):
                task.ignore_columns += undetermined_column_names
        del dataset_summary.meta_features_pp[ps_macros.STR_OTHER]
        # 2. columns with all null values
        if ps_macros.ALL_MISSING_PRESENCE in dataset_summary.meta_features_pp:
            column_names_with_all_missing_values = dataset_summary.meta_features_pp[ps_macros.ALL_MISSING_PRESENCE]
            if isinstance(column_names_with_all_missing_values, list):
                task.ignore_columns += column_names_with_all_missing_values
        del dataset_summary.meta_features_pp[ps_macros.ALL_MISSING_PRESENCE]

        labels = predict(task, dataset_summary)
        adapt = Adaptation(
            labels,
            task,
            dataset_summary,
            self.config,
        )
        pipelines = adapt.run_adaptation()

        return dataset, pipelines
