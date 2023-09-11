import tempfile
from pathlib import Path

import pandas as pd

from .executor import run
from .params import save_file
from .util.logging import setup_logger

logger = setup_logger()


class GeneratedModel:
    def __init__(self, input_dir, save_datasets_format, timeout, csv_encoding, csv_delimiter, params):
        self.files = dict()
        self.save_datasets_format = save_datasets_format
        self.timeout = timeout
        self.csv_encoding = csv_encoding
        self.csv_delimiter = csv_delimiter
        self.params = params

        self._readfile(input_dir / "final_train.py", input_dir)
        self._readfile(input_dir / "final_predict.py", input_dir)

        for filepath in input_dir.glob("lib/*.py"):
            self._readfile(filepath, input_dir)

        for filepath in input_dir.glob("**/*.pkl"):
            if save_datasets_format == "pickle" and "training.pkl" == filepath.name:
                continue
            self._readfile(filepath, input_dir)

    def _readfile(self, filepath, input_dir):
        with open(filepath, "rb") as f:
            self.files[str(filepath.relative_to(input_dir))] = f.read()

    def save(self, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "lib").mkdir(exist_ok=True)
        for filename, content in self.files.items():
            with open(output_dir / filename, "wb") as f:
                f.write(content)

    def fit(self, training_dataframe: pd.DataFrame):
        """
        Generate ML scripts for input data.

        Parameters
        ----------
        training_dataframe: pandas.DataFrame
            Training dataframe.

        Returns
        -------
        self: GeneratedModel
            GeneratedModel object itself
        """
        with tempfile.TemporaryDirectory() as temp_dir_path_str:
            temp_dir = Path(temp_dir_path_str).absolute()
            temp_dir.mkdir(exist_ok=True)
            filename = "training." + ("pkl" if self.save_datasets_format == "pickle" else "csv")
            save_file(training_dataframe, str(temp_dir / filename), self.csv_encoding, self.csv_delimiter)
            self.save(temp_dir)
            logger.info("Building model by generated pipeline...")
            result = run(str(temp_dir / "final_train.py"), self.timeout)
            if result.returncode != 0:
                raise RuntimeError(f"Training was failed due to the following Error: {result.error}")
            for filepath in temp_dir.glob("**/*.pkl"):
                if self.save_datasets_format == "pickle" and "training.pkl" == filepath.name:
                    continue
                self._readfile(filepath, temp_dir)
        return self

    def predict(self, test_dataframe: pd.DataFrame):
        """Predicts the output of the test_data and store in the prediction_result.csv.

        Parameters
        ---------
        test_dataframe: pd.DataFrame
            Dataframe used for predicting the result.

        Returns
        -------
        result_df : pd.DataFrame
            It returns the prediction_result.csv result in dataframe format.
        """
        with tempfile.TemporaryDirectory() as temp_dir_path_str:
            temp_dir = Path(temp_dir_path_str).absolute()
            temp_dir.mkdir(exist_ok=True)
            filename = "test." + ("pkl" if self.save_datasets_format == "pickle" else "csv")
            save_file(test_dataframe, str(temp_dir / filename), self.csv_encoding, self.csv_delimiter)
            self.save(temp_dir)
            logger.info("Predicting by built model...")
            result = run(str(temp_dir / "final_predict.py"), self.timeout)
            if result.returncode != 0:
                raise RuntimeError(f"Prediction was failed due to the following Error: {result.error}")
            result_df = pd.read_csv(temp_dir / "prediction_result.csv")
            return result_df
