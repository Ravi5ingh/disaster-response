import utility.util as u
import re as re
import nltk.tokenize as tkn
import nltk.corpus as co
import nltk.stem.porter as po
import nltk.stem.wordnet as wo
import pipetools as pt

def normalize_messages(disaster_df):
    """
    Given a disaster data frame, normalizes the message column
    :param disaster_df: The disaster data frame
    :return: The normalized data frame
    """

    disaster_df['message'] = disaster_df['message'].apply(lambda message:
                                                          (pt.pipe
                                                           | __normalize_text__
                                                           | __tokenize_text__
                                                           | __remove_stopwords__
                                                           # | __stem_text__
                                                           | __lemmatize_text__) (message))

    return disaster_df

def one_hot_encode_genre(disaster_df):
    """
    Perform one hot encoding on the 'genre' column
    :param disaster_df: The disaster data frame
    :return: One hot encoded disaster data frame
    """

    return u.one_hot_encode(disaster_df, 'genre')

def remove_columns(disaster_df):
    """
    Removes un-necessary columns from the disaster data frame
    :param disaster_df: The disaster data frame
    :return: A pruned data frame
    """

    return disaster_df.drop(['id', 'original', 'child_alone'], axis=1)

def normalize_related_category_values(disaster_df):
    """
    Replaces the '2' values in the 'related' column because they must be '1'
    :param disaster_df: The disaster data frame
    :return:
    """

    disaster_df.loc[disaster_df['related'] == 2, 'related'] = 1

    return disaster_df

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
