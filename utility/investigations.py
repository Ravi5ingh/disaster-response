from utility.util import *

def print_disaster_dupe_summary():
    """
    Goes through merged, and categorized disaster.csv and prints the ids that are duplicates and a preview of the
    messages
    """

    disaster = read_csv('data/disaster.csv')

    # Check for dupes
    ids = set()
    disaster['id'].apply(lambda x: ids.add(x))

    dupe_ids = []
    for id in ids:
        if row_count(disaster[disaster['id'] == id]) > 1:
            print(id)
            dupe_ids.append(id)

    for dupe_id in dupe_ids:
        print(disaster[disaster['id'] == dupe_id]['message'])

def print_unique_lengths_of_categories():
    """
    Prints all the different lengths that the 'categories' column has
    (If this print more than 1 number, than the data has a problem)
    """

    lengths = set()
    categories = read_csv('data/disaster_categories.csv')

    for index, row in categories.iterrows():

        lengths.add(len(row['categories'].split(';')))


    for length in lengths:
        print(length)

def say_hello():

    print('Hello World')