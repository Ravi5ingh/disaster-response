import models.investigation_code.investigations as inv
# import utility.util as ut
# import gensim

inv.show_disaster_pca_for('death')

# sentences = ['While it may not be obvious to everyone, there are a number of reasons creating random paragraphs can be useful'.split(' '),
#              'When a random word or a random sentence isnt quite enough, the next logical step is to find a random paragraph'.split(' '),
#              'We created the Random Paragraph Generator with you in mind'.split(' '),
#              'The process is quite simple'.split(' '),
#              'Choose the number of random paragraphs youd like to see and click the button'.split(' '),
#              'Your chosen number of paragraphs will instantly appear'.split(' '),
#              'While it may not be obvious to everyone, there are a number of reasons creating random paragraphs can be useful'.split(' '),
#              'A few examples of how some people use this generator are listed in the following paragraphs'.split(' ')]
#
# model = gensim.models.Word2Vec(sentences, min_count=1)

# disaster = ut.read_csv('../disaster.csv')
#
# disaster['message'] = disaster['message'].apply(lambda x: x.upper().split())
#
# model = gensim.models.Word2Vec(disaster['message'].array, min_count=5)
#
# model.save('../disaster.model')

# model = gensim.models.Word2Vec.load('../disaster.model')
#
# for word in model.wv.vocab:
#
#     print(word)
#     print(model[word])
#     input()

# for word, vector in model.wv.vocab.items():
#
#     print(word)
#     print(model[word])
#     input()

# ut.whats(model.wv.vocab)
#
# for vector in model.wv.vocab:
#
#     print(vector)
#
#     print(model[vector])
#
#     ut.whats(vector)
#
#     input()


# print(model['process'])
# print(model['quite'])