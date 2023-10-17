from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union

from .params import Code, Dataset, Task


class CodeBlockGenerator(ABC):
    def __init__(self, **kwargs):  # pragma: no cover
        pass

    @abstractmethod
    def generate_code(self, dataset: Dataset, task: Task) -> Tuple[Dataset, Code]:  # pragma: no cover
        pass


class PipelineGenerator(ABC):
    @abstractmethod
    def generate_pipeline(self, dataset: Dataset, task: Task):  # pragma: no cover
        pass

    @abstractmethod
    def save(self, output_dir: Union[Path, str]):  # pragma: no cover
        pass

    @abstractmethod
    def add_explanation(self):  # pragma: no cover
        pass
