import sys
import utility.util as ut
import pandas as pd
import os

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the data from the given locations, ready to be cleaned
    :param messages_filepath: The file path for the messages data
    :param categories_filepath: The file path for the categorization data
    :return: Merges the 2 sources on 'id' and returns the joined result
    """
    messages = ut.read_csv(messages_filepath)
    categories = ut.read_csv(categories_filepath)

    return pd.merge(messages, categories, on='id', how='inner')


def clean_data(disaster_df):
    """
    Split the categorization data into separate columns and remove duplicates
    :param disaster_df: The merged disaster data
    :return: Returns the cleaned data frame
    """

    # Check that the rows in the categories data all have 36 categories (If this doesn't print a 36, they don't)
    # print_unique_lengths_of_categories()

    # Get the column names
    column_names = [*map(lambda x: x[0:x.find('-')], disaster_df['categories'].iloc[0].split(';'))]

    # Split the categories and populate every new column
    cat_rows = disaster_df['categories'].apply(lambda x: x.split(';'))
    i = 0
    for column in column_names:
        disaster_df[column] = cat_rows.apply(lambda x: int(x[i][x[i].find('-') + 1:]))
        i += 1

    # Remove original categories column
    disaster_df = disaster_df.drop(columns=['categories'])

    # Remove Dupes
    # (Based on the dupe summary, dupe ids mean dupe rows)
    disaster_df = disaster_df.drop_duplicates(subset='id')

    # Check dupes (if this prints anything, there are)
    # print_disaster_dupe_summary()

    return disaster_df


def save_data(disaster_df, disaster_csv_filename, database_filename):
    """
    Save the cleaned disaster data frame as a csv and a DB file (DB file will be replaced if it exists)
    :param disaster_df: The cleaned disaster data frame
    :param disaster_csv_filename: The name of the csv file to save to
    :param database_filename: The name of the DB file to create
    """

    # Persist the cleaned data as a csv file
    disaster_df.to_csv(disaster_csv_filename, index=False)

    # If the DB file exists, delete it
    if os.path.exists(database_filename):
        os.remove(database_filename)

    # Save data to an sqlite db
    engine = create_engine('sqlite:///' + database_filename)
    disaster_df.to_sql(database_filename, engine, index=False)


def main():
    """
    Point of entry (Takes 4 to 5 arguments)
    """

    if len(sys.argv) == 5 or len(sys.argv) == 4:

        if len(sys.argv) == 5:
            messages_filepath, categories_filepath, database_filepath, disaster_csv_filename = sys.argv[1:]

        if len(sys.argv) == 4:
            messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
            disaster_csv_filename = 'disaster.csv'

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}, CSV FILE: {}'.format(database_filepath, disaster_csv_filename))
        save_data(df, disaster_csv_filename, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. Fourth argument (optional) is csv file '
              'name for output data \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db [disaster.csv]')


if __name__ == '__main__':
    main()