import logging

from datasets import DatasetDict, load_dataset
from transformers import set_seed


def load_data(name):
    logging.info("Start download file")
    DATASET_PATH = None
    DATASET_NAME = None
    split = "test"
    columns = None
    rename_columns = None
    if name == "humaneval":
        DATASET_PATH = "codeparrot/instructhumaneval"
        columns = ["task_id", "instruction"]
        rename_columns = ["instruction", "content"]
    elif name == "mbpp":
        DATASET_PATH = "mbpp"
        columns = ["task_id", "text"]
        rename_columns = ["text", "content"]
    elif name == "leetcode":
        DATASET_PATH = "greengerong/leetcode"
        split = "train"
        columns = ["id", "content"]
        rename_columns = ["id", "task_id"]
    else:
        Exception("Error in dataset name, can be only humaneval, mbpp, leetcode")
    dataset = load_dataset(path=DATASET_PATH, name=DATASET_NAME, cache_dir="./data/")[split].select_columns(columns)\
        .rename_column(rename_columns[0], rename_columns[1])
    logging.info(f"Finished download file, dataframe size: {len(dataset)}")
    return dataset


def preprocess_df(df):
    set_seed(0)
    train_test = df.train_test_split(test_size=0.2)
    test_validation = train_test["test"].train_test_split(test_size=0.5)
    train_test_validation = DatasetDict(
        {
            "train": train_test["train"],
            "test": test_validation["train"],
            "valid": test_validation["test"],
        }
    )
    return train_test_validation


if __name__ == "__main__":
    load_data("mbpp")
