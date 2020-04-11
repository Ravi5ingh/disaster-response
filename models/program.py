import utility.util as u
import models.investigations as inv

# inv.show_disaster_pca_for('fire')

# inv.find_most_biased_word_for('money')

# inv.create_word_bias_data('disaster.csv', 'bias.csv')

# inv.create_readble_bias('bias.csv', 'readable_bias.csv')

disaster = u.read_csv('disaster.csv')

# print(disaster['genre'].unique())

direct_count = u.row_count(disaster[disaster['genre']=='direct'])
social_count = u.row_count(disaster[disaster['genre']=='social'])
news_count = u.row_count(disaster[disaster['genre']=='news'])

print(direct_count)
print(social_count)
print(news_count)