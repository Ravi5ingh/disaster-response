import utility.util as u
import models.nl_processor as nlp
import models.investigations as inv
import ast

disaster = u.read_csv('../data/disaster.csv')\
    .pipe(nlp.remove_columns)\
    .pipe(nlp.one_hot_encode_genre)\
    .pipe(nlp.normalize_related_category_values)\
    .pipe(nlp.normalize_messages)

disaster.to_csv('disaster_tokenized.csv', index=False)

inv.create_word_bias_data('disaster_tokenized.csv', 'disaster_bias.csv')

inv.create_readble_bias('disaster_bias.csv', 'investigation_results/DisasterBias.db', 'DisasterBias')
