import sklearn.model_selection as ms
import sklearn.pipeline as pi
import sklearn.feature_extraction.text as te
import sklearn.ensemble as en
import sklearn.metrics as me
import numpy as np
import utility.util as ut
import pipetools as pt
import models.nl.processor as nl
import statistics as st

class SparseVecRandomForestPipeline:
    """
    This class represents the pipeline that houses models that work with sparse vector representations of the
    message column and trained with the Random Forest Classifier
    """

    def __init__(self, disaster_db_file, disaster_table_name):
        """
        .ctor
        :param disaster_db_file: The file path for the disaster db file of the data
        :param disaster_table_name: The name of the table inside the DB that stores the data
        """

        self.__disaster__ = ut.read_db(disaster_db_file, disaster_table_name)

        self.__categories_columns__ = ['related', 'request', 'offer',
                                  'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                                  'security', 'military', 'water', 'food', 'shelter',
                                  'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
                                  'infrastructure_related', 'transport', 'buildings', 'electricity',
                                  'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                                  'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                                  'other_weather', 'direct_report']

        self.__pipelines__ = {}

        self.__readable_summary__ = {}

        self.__accuracies__ = {}

    def init_fit_eval(self):
        """
        Initializes, fits, and evaluates a Random Forest Classifier on the disaster data
        """

        print('Creating Sparse Vector Pipeline...')

        X = self.__disaster__['message'].values

        i = 1
        num_cats = len(self.__categories_columns__)
        for category in self.__categories_columns__:

            Y = self.__disaster__[category]
            x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.25)

            current_pipeline = pi.Pipeline([
                ('vect', te.CountVectorizer(tokenizer=self.__tokenize_tweet__)),
                ('tfidf', te.TfidfTransformer()),
                ('clf', en.RandomForestClassifier())
            ])

            self.__pipelines__[category] = current_pipeline

            current_pipeline.fit(x_train, y_train)

            y_pred = current_pipeline.predict(x_test)

            self.__add_to_summary__(category, y_test, y_pred)

            print('Fitted ' + str(i) + ' out of ' + str(num_cats) + ' models')
            i += 1

        print('Done fitting. Model Evaluation Summary')
        for category in self.__readable_summary__:

            print('------' + str(category) + '------')
            print(self.__readable_summary__[category])

        print('Overall Accuracy: ' + str(st.mean(self.__accuracies__.values())))

    def predict(self, tweet):
        """
        Classify the tweet in terms of the categories concerned
        :param tweet: The raw tweet
        :return: The classifications for each concerned category
        """

        predictions = {}
        for category in self.__categories_columns__:
            predictions[category] = self.__pipelines__[category].predict([tweet])[0]

        return predictions


    #region Private

    def __tokenize_tweet__(self, tweet):
        """
        Take the raw tweet string and tokenize to a standardized string array
        :param tweet: The raw tweet
        :return: The tokenized tweet
        """

        return (pt.pipe
                | nl.normalize_text
                | nl.tokenize_text
                | nl.remove_stopwords
                | nl.lemmatize_text)(tweet)

    def __add_to_summary__(self, category_name, y_test, y_pred):
        """
        Add readable results to the summary based on the predicted vs actual values of 1 model
        :param y_test: The actual values
        :param y_pred: The predicted values
        """

        labels = np.unique(y_pred)
        confusion_matrix = me.confusion_matrix(y_test, y_pred, labels=labels)
        accuracy = (y_pred == y_test).mean()

        self.__readable_summary__[category_name] = ''
        self.__accuracies__[category_name] = accuracy

        self.__readable_summary__[category_name] += '\nLabels: ' + str(labels)
        self.__readable_summary__[category_name] += '\nConfusion Matrix:\n' + str(confusion_matrix)
        self.__readable_summary__[category_name] += '\nAccuracy: ' + str(accuracy)

    #endregion



