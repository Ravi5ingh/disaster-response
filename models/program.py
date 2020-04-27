import utility.util as u
import models.nl_processor as nlp
import models.investigations as inv
import ast
import numpy as np
import scipy.spatial as spa
import math
import gensim as gs
import sklearn.feature_extraction.text as te
import models.pipelines as pi
import models.sparsevec_randomforest_pipeline as sp
import models.test_obj as tt
import statistics as st


pipeline = sp.SparseVecRandomForestPipeline('../data/disaster.csv')

pipeline.init_fit_eval()

u.to_pkl(pipeline, 'investigation_results/random_forest/test.pkl')
