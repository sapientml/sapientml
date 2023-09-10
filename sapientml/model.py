import tempfile
from pathlib import Path

import pandas as pd

from .executor import run
from .params import Dataset
from .util.logging import setup_logger

logger = setup_logger()


class GeneratedModel:
    def __init__(self, output_dir, save_datasets_format, timeout, csv_encoding, csv_delimiter, params):
        self.files = dict()
        self.save_datasets_format = save_datasets_format
        self.timeout = timeout
        self.csv_encoding = csv_encoding
        self.csv_delimiter = csv_delimiter
        self.params = params

        self._readfile(output_dir / "final_train.py", output_dir)
        self._readfile(output_dir / "final_predict.py", output_dir)

        for filepath in output_dir.glob("lib/*.py"):
            self._readfile(filepath, output_dir)

        for filepath in output_dir.glob("**/*.pkl"):
            if save_datasets_format == "pickle" and "training.pkl" == filepath.name:
                continue
            self._readfile(filepath, output_dir)

    def _readfile(self, filepath, output_dir):
        with open(filepath, "rb") as f:
            self.files[str(filepath.relative_to(output_dir))] = f.read()

    def _writefiles(self, output_dir):
        (output_dir / "lib").mkdir(exist_ok=True)
        for filename, content in self.files.items():
            with open(output_dir / filename, "wb") as f:
                f.write(content)

    def fit(self, training_data):
        with tempfile.TemporaryDirectory() as temp_dir_path_str:
            temp_dir = Path(temp_dir_path_str).absolute()
            temp_dir.mkdir(exist_ok=True)
            Dataset(
                training_data=training_data,
                save_datasets_format=self.save_datasets_format,
                csv_encoding=self.csv_encoding,
                output_dir=temp_dir,
            )
            self._writefiles(temp_dir)
            logger.info("Building model by generated pipeline...")
            result = run(str(temp_dir / "final_train.py"), self.timeout)
            if result.returncode != 0:
                raise RuntimeError(f"Training was failed due to the following Error: {result.error}")
            for filepath in temp_dir.glob("**/*.pkl"):
                if self.save_datasets_format == "pickle" and "training.pkl" == filepath.name:
                    continue
                self._readfile(filepath, temp_dir)
        return self

    def predict(self, test_data):
        with tempfile.TemporaryDirectory() as temp_dir_path_str:
            temp_dir = Path(temp_dir_path_str).absolute()
            temp_dir.mkdir(exist_ok=True)
            Dataset(
                test_data=test_data,
                save_datasets_format=self.save_datasets_format,
                csv_encoding=self.csv_encoding,
                output_dir=temp_dir,
            )

            self._writefiles(temp_dir)

            logger.info("Predicting by built model...")
            result = run(str(temp_dir / "final_predict.py"), self.timeout)
            if result.returncode != 0:
                raise RuntimeError(f"Prediction was failed due to the following Error: {result.error}")
            result_df = pd.read_csv(temp_dir / "prediction_result.csv")
            return result_df
