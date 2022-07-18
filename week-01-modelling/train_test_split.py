import logging
import os

import pandas as pd
from src.dataset_splitter import stratify_shuffle_split_subsets


def split_and_save_datasets(df: pd.DataFrame, save_path: str):
    logging.info(f"Original dataset: {len(df)}")
    df = df.drop_duplicates()
    df = df.drop(["Genre", "Reality-TV"], axis=1)
    logging.info(f"Final dataset: {len(df)}")

    train_df, valid_df, test_df = stratify_shuffle_split_subsets(
        df,
        img_path_column="Id",
        train_fraction=0.8,
        verbose=True,
    )
    logging.info(f"Train dataset: {len(train_df)}")
    logging.info(f"Valid dataset: {len(valid_df)}")
    logging.info(f"Test dataset: {len(test_df)}")

    train_df.to_csv(os.path.join(save_path, "train_df.csv"), index=False)
    valid_df.to_csv(os.path.join(save_path, "valid_df.csv"), index=False)
    test_df.to_csv(os.path.join(save_path, "test_df.csv"), index=False)
    logging.info("Datasets successfully saved!")


if __name__ == "__main__":
    save_path = os.path.join(os.environ.get("ROOT_PATH"))
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv(os.path.join(os.path.join(os.environ.get("ROOT_PATH")), "train.csv"))
    split_and_save_datasets(df, save_path)
