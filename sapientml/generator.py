from abc import ABC, abstractmethod
from typing import Tuple

from .params import Code, Config, Dataset, Task


class CodeBlockGenerator(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def generate_code(self, dataset: Dataset, task: Task) -> Tuple[Dataset, Code]:
        pass


class PipelineGenerator(ABC):
    @abstractmethod
    def generate_pipeline(self, dataset: Dataset, task: Task) -> list[Code]:
        pass
