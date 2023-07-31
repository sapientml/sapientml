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
import re
from pathlib import Path

import fasttext
import MeCab
import numpy as np
import pandas as pd
import requests
from jinja2 import Environment, FileSystemLoader
from pandas.api.types import infer_dtype
from sapientml.params import Pipeline, Task
from sapientml.util.logging import setup_logger

logger = setup_logger()

INHIBITED_SYMBOL_PATTERN = re.compile(r"[\{\}\[\]\",:<'\\]+")


template_env = Environment(
    loader=FileSystemLoader(f"{os.path.dirname(__file__)}/../../../templates/rule_based"), trim_blocks=True
)


def _is_strnum_column(c):
    c2 = c.loc[c.notnull()]
    c2 = pd.to_numeric(c2, errors="coerce")
    ratio = c2.notnull().sum() / c2.shape[0]
    return ratio > 0.9


def _is_category_column(c):
    c2 = c.loc[c.notnull()]
    num = c2.nunique()
    if num <= 2:
        return "binary"
    elif num <= 20 and (num / c2.shape[0] < 0.3):
        return "small"
    elif c2.value_counts(normalize=True).head(20).sum() > 0.8:
        return "large"
    else:
        return False


def _render(tpl, *args, **kwargs):
    code = tpl.render(*args, **kwargs)
    return "\n".join([line for line in code.split("\n") if len(line) > 0]) + "\n\n"


def check_column_language(df: pd.DataFrame) -> list[str]:
    # suppress meaningless warning when model loading
    os.makedirs(Path(os.path.dirname(__file__)) / "lib", exist_ok=True)
    model_path = Path(os.path.dirname(__file__)) / "lib" / "lid.176.bin"
    if not model_path.exists():
        response = requests.get("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
        response.raise_for_status()  # check download status and raise if download is failed
        with open(model_path, mode="wb") as f:
            f.write(response.content)
    fasttext.FastText.eprint = lambda x: None
    model = fasttext.load_model(model_path._str)
    object_columns = df.select_dtypes("object").columns
    col_language_dict = {}
    for col in object_columns:
        if not _is_category_column(df[col]):
            c2 = df[col][df[col].notnull()]
            if c2.shape[0] > 1000:
                c2 = c2.sample(1000, random_state=17)
            value_languages = (
                c2.astype(str)
                .str.replace("\n", "")
                .str.replace("\r", "")
                .apply(lambda x: model.predict(x)[0][0].split("__label__")[1])
            )
            col_language_dict[col] = value_languages.mode()[0]
    return col_language_dict


def tokenize(text, use_pos_list, use_word_stemming, tokenizer):
    node = tokenizer.parseToNode(text)
    terms = []
    while node:
        features = node.feature.split(",")
        pos = features[0]
        if pos != "BOS/EOS":
            if use_word_stemming:
                term = features[6]
                if (pos == "名詞") & (features[1] == "数"):
                    term = node.surface
            else:
                term = node.surface
            if use_pos_list:
                if pos in use_pos_list:
                    terms.append(term)
            else:
                terms.append(term)
        node = node.next
    return " ".join(terms)


def check_cols_has_symbols(columns: list) -> list[str]:
    cols_has_symbols = []
    for col in columns:
        if INHIBITED_SYMBOL_PATTERN.search(col):
            cols_has_symbols.append(col)
    return cols_has_symbols


def remove_symbols(column_name: str) -> str:
    return INHIBITED_SYMBOL_PATTERN.sub("", column_name)


def generate_code_rule_based(df: pd.DataFrame, task: Task):
    code_validation = code_test = code_train = code_predict = ""

    # Remove special symbols that interfere with visualization and model training
    cols_has_symbols = []
    cols_has_symbols = check_cols_has_symbols(df.columns.to_list())
    if cols_has_symbols:
        logger.warning(
            f"Symbols that inhibit training and visualization will be removed from column name {str(cols_has_symbols)}."
        )
        df = df.rename(columns=lambda col: remove_symbols(col) if col in cols_has_symbols else col)
        task.target_columns = [remove_symbols(col) if col in cols_has_symbols else col for col in task.target_columns]
        tpl = template_env.get_template("rename_columns.py.jinja")
        code_validation += _render(tpl, training=True, test=True, cols_has_symbols=cols_has_symbols)
        code_test += _render(tpl, training=True, test=True, cols_has_symbols=cols_has_symbols)
        code_train += _render(tpl, training=True, test=False, cols_has_symbols=cols_has_symbols)
        code_predict += _render(tpl, training=False, test=True, cols_has_symbols=cols_has_symbols)

    # handle list(tuple, dict) value in dataframe.
    # in generated scripts, visualisation will be executed before pre-processing such as handle mixed-type.
    # so, need to check before mixed-type column names are added to suppress errors during visualisation.
    cols_iterable_values = []
    for col in df.columns:
        exist_list_values = [x for x in df[col] if type(x) in [list, tuple, dict]]
        if len(exist_list_values) > 0:
            cols_iterable_values.append(col)
            df[col] = df[col].fillna("").astype(str)
    if cols_iterable_values:
        tpl = template_env.get_template("handle_iterable_values.py.jinja")
        code_validation += _render(tpl, training=True, test=True, cols_iterable_values=cols_iterable_values)
        code_test += _render(tpl, training=True, test=True, cols_iterable_values=cols_iterable_values)
        code_train += _render(tpl, training=True, test=False, cols_iterable_values=cols_iterable_values)
        code_predict += _render(tpl, training=False, test=True, cols_iterable_values=cols_iterable_values)

    # handle mixed-type columns
    # split a columns into 2 columns, one column has only numeric, another columns has only string
    # this operation should be done before calculating meta features
    dtypes = df.drop(task.target_columns, axis=1).apply(infer_dtype)
    is_strnum = df.drop(task.target_columns, axis=1).apply(_is_strnum_column)
    mix_typed_cols = dtypes[
        dtypes.str.fullmatch("|".join(["mixed", "mixed-integer"])) | (dtypes.str.fullmatch("string") & is_strnum)
    ].index.to_list()
    cols_numeric_and_string = []
    for col in mix_typed_cols:
        cols_numeric_and_string.append(col)
        only_str = col + "__str"
        only_num = col + "__num"
        df[only_str] = np.where(pd.to_numeric(df[col], errors="coerce").isnull(), df[col], np.nan)
        # without .astype(float), cannot recongnize as `int` or `float`, leading to generate inappropriate code snippet
        df[only_num] = np.where(pd.to_numeric(df[col], errors="coerce").isnull(), np.nan, df[col]).astype(float)
        df = df.drop(col, axis=1)
    if cols_numeric_and_string:
        tpl = template_env.get_template("handle_mixed_typed_columns.py.jinja")
        code_validation += _render(tpl, training=True, test=True, cols_numeric_and_string=cols_numeric_and_string)
        code_test += _render(tpl, training=True, test=True, cols_numeric_and_string=cols_numeric_and_string)
        code_train += _render(tpl, training=True, test=False, cols_numeric_and_string=cols_numeric_and_string)
        code_predict += _render(tpl, training=False, test=True, cols_numeric_and_string=cols_numeric_and_string)

    # meta features must be calculated after replacing inf with nan,
    # becuase the replaced nan must be preprocessed in the generated code.
    # handling inf as nan must be AFTER to_csv() to keep dataset files intact
    inf_bool_s = df.drop(task.target_columns, axis=1).isin([np.inf, -np.inf]).any()
    cols_inf_values = inf_bool_s.index[inf_bool_s].tolist()
    if cols_inf_values:
        df[cols_inf_values] = df[cols_inf_values].replace([np.inf, -np.inf], np.nan)
        tpl = template_env.get_template("handle_inf_columns.py.jinja")
        code_validation += _render(tpl, training=True, test=True, cols_inf_values=cols_inf_values)
        code_test += _render(tpl, training=True, test=True, cols_inf_values=cols_inf_values)
        code_train += _render(tpl, training=True, test=False, cols_inf_values=cols_inf_values)
        code_predict += _render(tpl, training=False, test=True, cols_inf_values=cols_inf_values)

    # handle Japanese text columns
    cols_japanese_text = []
    column_language_dict = check_column_language(df.drop(task.target_columns, axis=1))
    cols_japanese_text = [key for key, val in column_language_dict.items() if val == "ja"]
    if cols_japanese_text:
        tokenizer = MeCab.Tagger()
        for col in cols_japanese_text:
            df[col] = df[col].fillna("")
            df[col] = df[col].apply(lambda x: tokenize(x, task.use_pos_list, task.use_word_stemming, tokenizer))
        tpl = template_env.get_template("handle_japanese_text.py.jinja")
        code_validation += _render(tpl, task=task, training=True, test=True, cols_japanese_text=cols_japanese_text)
        code_test += _render(tpl, task=task, training=True, test=True, cols_japanese_text=cols_japanese_text)
        code_train += _render(tpl, task=task, training=True, test=False, cols_japanese_text=cols_japanese_text)
        code_predict += _render(tpl, task=task, training=False, test=True, cols_japanese_text=cols_japanese_text)

    return Pipeline(
        code_for_validation=code_validation,
        code_for_test=code_test,
        code_for_train=code_train,
        code_for_predict=code_predict,
    )
