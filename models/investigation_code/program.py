import utility.util as ut
import global_variables as gl
import sklearn.feature_extraction.text as te
import models.pipelines.neural_word2vec_pipeline as pn
import models.nl.processor as nl
import pipetools as pt
import models.investigation_code.investigations as iv
import sklearn.metrics as me
import ast
import numpy as np



y1 = [
    [1,1,0],
    [1,1,0],
    [1,0,0],
    [0,1,0]
]

for e in y1:
    print(e)

y1 = np.array(y1).T

print('----')
for e in y1:
    print(e)

# X = np.array(X).T.tolist()

# model = pn.NeuralWord2VecPipeline('../../data/DisasterResponse.db', 'Disaster')
#
# model.init_fit_eval()
#
# ut.to_pkl(model, 'nn_model.pkl')

# def __tokenize_tweet__(tweet):
#     """
#     Take the raw tweet string and tokenize to a standardized string array
#     :param tweet: The raw tweet
#     :return: The tokenized tweet
#     """
#
#     return (pt.pipe
#             | nl.normalize_text
#             | nl.tokenize_text
#             | nl.remove_stopwords
#             | nl.lemmatize_text)(tweet)
#
# ss = [
#     'Hello my name is Ravi',
#     'My name is Ravi',
#     'YOLO YOLO'
# ]
#
# cc = te.CountVectorizer(tokenizer=__tokenize_tweet__)
#
# print(cc.fit_transform(ss))


# [
#     'Hello my name is Ravi',
#     'My name is Ravi',
#     'YOLO YOLO'
# ]
#
# goes to...
#
# [[1 1 1 1 1 0]
#  [0 1 1 1 1 0]
#  [0 0 0 0 0 2]]