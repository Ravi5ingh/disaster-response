import utility.util as u
import models.nl_processor as nlp

disaster = u.read_csv('../data/disaster.csv')\
    .pipe(nlp.remove_columns)\
    .pipe(nlp.one_hot_encode_genre)\
    .pipe(nlp.normalize_related_category_values)\
    .pipe(nlp.normalize_messages)

