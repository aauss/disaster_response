import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """Loads message and category data and merges data.

    Args:
        messages_filepath: Filepath fo message.csv
        categories_filepath: Filepath fo categories.csv

    Returns:
        Merged categories and message data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on="id")


def clean_data(df: pd.DataFrame):
    """Cleans dataframe.

    Args:
        df: Merged dataframe to clean.

    Returns:
        Cleaned dataframe.
    """
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]

    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])
        categories[column] = [1 if i > 0 else 0 for i in categories[column]]

    df = df.drop("categories", axis=1)
    df = pd.concat([df, categories], axis=1)
    return df.drop_duplicates()


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """Saves dataframne to a sql database.

    Args:
        df: Dataframe to save.
        database_filename: Path to database.
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql(Path(database_filename).stem, engine, index=False, if_exists="replace")


def main() -> None:
    """Loads, preprocesses, and saves data for machine learning model.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
