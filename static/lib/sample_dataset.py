from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
from sklearn.model_selection import train_test_split


def _sampled_training(dev_training_dataset, train_size, stratify, task_type) -> pd.DataFrame:
    sampled_training_dataset, _ = train_test_split(
        dev_training_dataset,
        train_size=train_size,
        stratify=stratify if task_type == "classification" else None,
    )
    return sampled_training_dataset  # type: ignore


def sample_dataset(
    dataframe: pd.DataFrame,
    sample_size: int,
    target_columns: list[str],
    task_type: str,
) -> pd.DataFrame:
    # Sample the training set if the dataset is big
    # FIXME
    sampled_training_dataset = None
    num_of_rows = len(dataframe.index)
    if num_of_rows >= sample_size:
        rare_labels = []
        dataframe_alltargets = None
        if task_type == "classification":
            dataframe_alltargets = dataframe[target_columns].astype(str).apply("".join, axis=1)
            label_count = dataframe_alltargets.value_counts()
            rare_labels = label_count.loc[label_count == 1].index.tolist()

        if rare_labels and dataframe_alltargets is not None:
            dataframe_rare = dataframe[dataframe_alltargets.isin(rare_labels)]
            rare_index = dataframe_rare.index.values

            dataframe_wo_rare = dataframe.drop(rare_index)

            num_of_labels = [len(dataframe_wo_rare[target].value_counts()) for target in target_columns]

            rare_to_all_ratio = int(
                Decimal(sample_size * len(dataframe_rare) / len(dataframe)).quantize(
                    Decimal("0"), rounding=ROUND_HALF_UP
                )
            )
            not_rare_to_all_ratio = int(
                Decimal(sample_size * len(dataframe_wo_rare) / len(dataframe)).quantize(
                    Decimal("0"), rounding=ROUND_HALF_UP
                )
            )

            stratify_wo_rare = None

            if len(dataframe_rare) == len(dataframe):
                sampled_training_dataset = _sampled_training(dataframe, sample_size, None, task_type)

            elif rare_to_all_ratio in [0, 1]:
                sampled_training_dataset_rare = dataframe_rare

                if max(num_of_labels) >= sample_size:
                    stratify_wo_rare = None
                else:
                    stratify_wo_rare = dataframe_wo_rare[target_columns]
                sampled_training_dataset_wo_rare = _sampled_training(
                    dataframe_wo_rare,
                    sample_size - len(sampled_training_dataset_rare),
                    stratify_wo_rare,
                    task_type,
                )

                sampled_training_dataset = pd.concat(
                    [sampled_training_dataset_wo_rare, sampled_training_dataset_rare]  # type: ignore
                )

            elif not_rare_to_all_ratio in [0, 1]:
                sampled_training_dataset_wo_rare = dataframe_wo_rare
                sampled_training_dataset_rare = _sampled_training(
                    dataframe_rare,
                    sample_size - len(sampled_training_dataset_wo_rare),
                    None,
                    task_type,
                )

                sampled_training_dataset = pd.concat(
                    [sampled_training_dataset_wo_rare, sampled_training_dataset_rare]  # type: ignore
                )

            else:
                if max(num_of_labels) >= sample_size:
                    stratify_wo_rare = None
                else:
                    stratify_wo_rare = dataframe_wo_rare[target_columns]

                sampled_training_dataset_wo_rare = _sampled_training(
                    dataframe_wo_rare, not_rare_to_all_ratio, stratify_wo_rare, task_type
                )
                sampled_training_dataset_rare = _sampled_training(dataframe_rare, rare_to_all_ratio, None, task_type)

                sampled_training_dataset = pd.concat(
                    [sampled_training_dataset_wo_rare, sampled_training_dataset_rare]  # type: ignore
                )

        else:
            num_of_labels = [len(dataframe[target].value_counts()) for target in target_columns]
            if max(num_of_labels) >= sample_size:
                stratify_wo_rare = None
            else:
                stratify_wo_rare = dataframe[target_columns]

            sampled_training_dataset = _sampled_training(dataframe, sample_size, stratify_wo_rare, task_type)
        return sampled_training_dataset
    else:
        return dataframe
