from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union

from .params import Code, Dataset, Task


class CodeBlockGenerator(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def generate_code(self, dataset: Dataset, task: Task) -> Tuple[Dataset, Code]:
        pass


class PipelineGenerator(ABC):
    @abstractmethod
    def generate_pipeline(self, dataset: Dataset, task: Task):
        pass

    @abstractmethod
    def save(self, output_dir: Union[Path, str]):
        pass

    @abstractmethod
    def add_explanation(self):
        pass
