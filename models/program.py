import utility.util as u
import models.nl_processor as nlp
import models.investigations as inv
import ast
import numpy as np
import scipy.spatial as spa
import math
import gensim as gs

# u.download_file('https://drive.google.com/uc?id=1Wh_uQ62XHruBSDSA5Ujb-eY6OcOLDRW-&export=download', 'word2vec/word2vec_cache.pkl')

v, s = u.try_word2vec('asddss')
print(v)

# file_id = '1kzCpXqZ_EILFAfK4G96QZBrjtezxjMiO'
# destination = 'word2vec/GoogleWord2VecModel.bin'
# print('Downloading...')
# u.download_file_from_google_drive(file_id, destination)
# print('Done')

# file = 'D:\Ravi\workspace\disaster-response\models\word2vec\GoogleWord2VecModel_downloaded.bin'
# google_word2vec_model = gs.models.KeyedVectors.load_word2vec_format(file, binary=True)
# print(google_word2vec_model['fire'])

# cache = u.read_pkl('word2vec/word2vec_cache.pkl')
# print(cache['fire'])


# disaster_df = u.read_csv('investigation_results/try_nn/disaster_normalized.csv')
# disaster_df['message'] = disaster_df['message'].apply(ast.literal_eval)

# inv.try_nn_avgvec_with(disaster_df, 'weather_related', 'deleteme.pkl')
#
# inv.show_disaster_pca_avgvec(disaster_df, 'search_and_rescue')
#
# inv.pca_compare_categories(disaster_df, 'shelter', 'transport')