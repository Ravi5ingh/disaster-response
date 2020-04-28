import re
import nltk.corpus as co
import nltk.stem.porter as po
import nltk.stem.wordnet as wo
import nltk.tokenize as tkn
import utility.util as ut
import sklearn.feature_extraction.text as st
import sklearn.ensemble as en
import sklearn.model_selection as ms
import sklearn.metrics as me
import sklearn.pipeline as pi
import numpy as np
import pipetools as pt

def create_disaster_pipeline(disaster_csv_path, category_name):

    disaster = ut.read_csv(disaster_csv_path)

    print('Getting data...')
    X = disaster['message'].values
    Y = disaster[category_name].values
    x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.3)

    print('Creating pipeline...')
    pipeline = pi.Pipeline([
        ('vect', st.CountVectorizer(tokenizer = lambda text: (pt.pipe
                                                              | __normalize_text__
                                                              | __tokenize_text__
                                                              | __remove_stopwords__
                                                              | __lemmatize_text__)(text))),
        ('tfidf', st.TfidfTransformer()),
        ('clf', en.RandomForestClassifier())
    ])

    print('Fitting pipeline...')
    pipeline.fit(x_train, y_train)

    print('Predicting with pipeline...')
    y_pred = pipeline.predict(x_test)

    print('Displaying results...')
    display_results(y_test, y_pred)

    pass

def create_disaster_sequence(disaster_csv_path, category_name):

    disaster = ut.read_csv(disaster_csv_path)

    print('Getting Data...')
    X = disaster['message'].values
    Y = disaster[category_name].values
    x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.3)

    print('Tokenizing and count vectorizing...')
    vect = st.CountVectorizer(tokenizer= lambda message: (pt.pipe
                                                           | __normalize_text__
                                                           | __tokenize_text__
                                                           | __remove_stopwords__
                                                           # | __stem_text__
                                                           | __lemmatize_text__) (message))

    print('Tfidf transforming...')
    tfidf = st.TfidfTransformer()
    classifier = en.RandomForestClassifier()

    print('Fitting classifier on train...')
    x_train_counts = vect.fit_transform(x_train)
    x_train_tfidf = tfidf.fit_transform(x_train_counts)
    classifier.fit(x_train_tfidf, y_train)

    print('Running classifier on test...')
    x_test_counts = vect.transform(x_test)
    x_test_tfidf = tfidf.transform(x_test_counts)
    y_pred = classifier.predict(x_test_tfidf)

    print('Displaying results...')
    display_results(y_test, y_pred)

def display_results(y_test, y_pred):

    labels = np.unique(y_pred)
    confusion_mat = me.confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)


# region Private

def __normalize_text__(text, num_value='num_value'):
    """
    Normalize a message for analysis (ie. convert to lower case alpha numeric text)
    :param text: The input text
    :param num_value: The default replacement for a numeric value
    :return: The normalized text
    """

    return re.sub(r'[^a-zA-Z ]+', num_value, re.sub(r'[^a-zA-Z0-9 ]', '', text.lower()))

def __tokenize_text__(text):
    """
    Splits text into an array of tokens
    :param text: The input text
    :return: The tokenized text
    """

    return tkn.word_tokenize(text)

def __remove_stopwords__(tokenized_text):
    """
    Removes stopwords from a piece of text
    :param tokenized_text: The input text array of tokens
    :return: The output text array of tokens without the stopwords
    """

    return [token for token in tokenized_text if token not in __stop_words__]

def __stem_text__(tokenized_text):
    """
    Stems all tokens in the input tokenized text
    :param tokenized_text: The tokenized text
    :return: The tokenized text with stemmed words
    """

    return [__stemmer__.stem(token) for token in tokenized_text]

def __lemmatize_text__(tokenized_text):
    """
    Lemmatizes all tokens in the input tokenized text
    :param tokenized_text: The tokenized text
    :return: The tokenized text with lemmatized words
    """

    return [__lemmatizer__.lemmatize(token) for token in tokenized_text]

# Locally initialized stop words (optimization)
__stop_words__ = co.stopwords.words('english')

# Locally initialized stemmer (optimization)
__stemmer__ = po.PorterStemmer()

# Locally initialized lemmatizer (optimization)
__lemmatizer__ = wo.WordNetLemmatizer()

#endregion

