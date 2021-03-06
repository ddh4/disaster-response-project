"""process_data.py docstring.

process_data.py is the ETL process for extracting data from two data sources,
transforming the data by extracting categories into individual features and finally,
the merged data is saved as an SQLite database for use in train_classifier.py.

This script takes 3 arguments: a filepath to the messages data, a filepath to the,
categories dataset and a filepath for the resulting SQLite database.

Example:
    An example of using this script is below::

        $ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
"""

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function loads the data from two sources.

    Args:
        messages_filepath: a string path.
        categories_filepath: a string path.

    Returns:
        A merged dataframe containing both datasets.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return pd.merge(messages, categories, on='id', how='left')

def clean_data(df):
    """
    This function transforms the merged dataset and removes duplicates.

    Args:
        df: the dataframe containing the merged data to be cleansed.

    Returns:
        A cleansed dataframe containing no duplicates and categories as
        individual features.
    """
    categories = df.categories.str.split(pat=';', expand=True)
    row = categories.iloc[1, :]
    category_colnames = [cat[:-2] for cat in row.values]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # include 36 new categories
    df = pd.concat([df, categories], axis=1)

    # remove erroneous values in related (non-binary column containing 2)
    df = df[df['related'] != 2]

    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)

    return df

def save_data(df, database_filename):
    """
    This function saves the dataset to an SQLite database.

    Args:
        df: the dataframe to be saved.
        database_filename: the path to create the SQLite database.

    Returns:
        None.
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Messages', engine, index=False, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        # parse command line arguments
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
