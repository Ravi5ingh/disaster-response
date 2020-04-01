from investigations import *
from sqlalchemy import create_engine

# Check that the rows in the categories data all have 36 categories (If this doesn't print a 36, they don't)
# print_unique_lengths_of_categories()

# Merge Data
messages = read_csv('data/disaster_messages.csv')
categories = read_csv('data/disaster_categories.csv')
disaster = pd.merge(messages, categories, on='id', how='inner')

# Split categories column
column_names = [*map(lambda x: x[0:x.find('-')], disaster['categories'].iloc[0].split(';'))]

cat_rows = disaster['categories'].apply(lambda x: x.split(';'))

i = 0
for column in column_names:
    disaster[column] = cat_rows.apply(lambda x: int(x[i][x[i].find('-') + 1:]))
    i += 1

# Remove original categories column
disaster = disaster.drop(columns=['categories'])

# Remove Dupes
# (Based on the dupe summary, dupe ids mean dupe rows)
disaster = disaster.drop_duplicates(subset='id')

# Persist
disaster.to_csv('data/disaster.csv', index=False)

# Save data to an sqlite db
engine = create_engine('sqlite:///DisasterResponse.db')
disaster.to_sql('Disaster', engine, index=False)

# Check dupes (if this prints anything, there are)
# print_disaster_dupe_summary()
