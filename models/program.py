import utility.util as u
import models.investigations as inv
import models.nl_processor as nlp
import pandas as pd
# import nltk.tokenize as tkn
import nltk

# disaster = u.read_csv('../data/disaster.csv')\
#     .pipe(nlp.remove_columns)\
#     .pipe(nlp.one_hot_encode_genre)\
#     .pipe(nlp.normalize_related_category_values)\
#     .pipe(nlp.normalize_messages)\
#     .pipe(nlp.tokenize_messages)

nltk.download()

# sen = 'hello i am very good boy'
#
# ss = tkn.word_tokenize(sen)
#
# print(ss)