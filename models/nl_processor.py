import utility.util as u
import pandas as pd
import re as re
import nltk.tokenize as tkn

def lemmatize_messages(disaster_df):
    """
    Lemmatize all words in the message column of the data frame
    :param disaster_df: The disaster data frame
    :return: The data frame with lemmatized values
    """

    return disaster_df

def stem_messages(disaster_df):
    """
    Stem all the words in the message column of the data frame
    :param disaster_df: The disaster data frame
    :return: The data frame with stemmed values
    """

def remove_stopwords_messages(disaster_df):
    """
    Removes stop words from the message column in the data frame
    :param disaster_df: The disaster data frame
    :return: The processed data frame
    """

    return disaster_df

def tokenize_messages(disaster_df):
    """
    Given a disaster data frame, tokenize the message column
    :param disaster_df: The disaster data frame
    :return: The tokenized data frame
    """

    # disaster_df['message'] = disaster_df['message'].apply(lambda x: tkn.word_tokenize(x))
    #
    # disaster_df.to_csv('test.csv', index=False)
    #
    # print(disaster_df)

    return disaster_df


def normalize_messages(disaster_df):
    """
    Given a disaaster data frame, normalizes the message column
    :param disaster_df: The disaster data frame
    :return: The normalized data frame
    """

    disaster_df['message'] = disaster_df['message'].apply(u.to_lower_alpha_numeric)

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
