import os

from datasets import load_dataset

from flaird.utils.arguments import DataArguments

os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"


def load_flaird_dataset(data_args: DataArguments):
    train_dataset = load_dataset(
        path=data_args.dataset_train_path,
        name=data_args.dataset_train_config_name,
        cache_dir=data_args.dataset_cache_dir,
    )
    train_dataset = train_dataset[list(train_dataset.keys())[0]]
    train_dataset = train_dataset.select_columns(
        ["text", "label", "generator_label", "forensic_features"]
    )
    validation_dataset = load_dataset(
        path=data_args.dataset_validation_path,
        name=data_args.dataset_validation_config_name,
        cache_dir=data_args.dataset_cache_dir,
    )
    validation_dataset = validation_dataset[list(validation_dataset.keys())[0]]
    validation_dataset = validation_dataset.select_columns(
        ["text", "label", "generator_label", "forensic_features"]
    )

    if data_args.shuffle_train_dataset:
        train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(data_args.max_train_samples, len(train_dataset)))
        )
    if data_args.max_eval_samples is not None:
        validation_dataset = validation_dataset.select(
            range(min(data_args.max_eval_samples, len(validation_dataset)))
        )

    return train_dataset, validation_dataset
