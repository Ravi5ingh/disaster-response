import gensim
import string

import utility.util as u
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.spatial as spa
import pickle as pk
import sklearn.neural_network as nn
import sklearn.model_selection as ms
import sklearn.metrics as me

import ast
import os
import math
import models.nl_processor as nlp

from sklearn.decomposition import PCA
from sklearn import svm
from itertools import *

def show_disaster_tsne(disaster_df, category_name):
    """
    Perform t-SNE dimensionality reduction on the average of the message word vectors and labels the cluster based
    on the category name
    :param disaster_df: The NORMALIZED disaster df
    :param category_name: The category to use for labelling
    """

    word_frequency = get_disaster_word_frequency(disaster_df)

    disaster_df = disaster_df.sample(10000)

    total = u.row_count(disaster_df)

    words = []
    X = []
    Y = []
    for index, row in disaster_df.iterrows():

        message_vectors = []
        for word in row['message']:
            # Disregards less than 50 instances
            if word_frequency[word] < 50:
                continue

            # Process the rest
            vector, op_success = u.try_word2vec(word)
            if op_success:
                words.append(word)
                message_vectors.append(vector)

        if message_vectors:
            X.append(np.average(message_vectors, axis=0))
            Y.append(row[category_name])

        if index%10000==0:
            print('Done ' + str(index) + ' of ' + str(total))

    u.show_2d_tsne(X, Y, ['r', 'g'])

def pca_compare_categories(disaster_df, category_zero, category_one):
    """
    Show 2D PCA to contrast 2 categories in the disaster df
    :param disaster_df: The NORMALIZED disaster df
    :param category_zero: The first category
    :param category_one: The second category
    """

    word_frequency = get_disaster_word_frequency(disaster_df)

    total = u.row_count(disaster_df)

    words = []
    X = []
    Y = []
    for index, row in disaster_df.iterrows():

        message_vectors = []
        for word in row['message']:
            # Disregards less than 50 instances
            if word_frequency[word] < 50:
                continue

            # Process the rest
            vector, op_success = u.try_word2vec(word)
            if op_success:
                words.append(word)
                message_vectors.append(vector)

        if message_vectors:
            X.append(np.average(message_vectors, axis=0))
            if row[category_zero] == 0 and row[category_one] == 0:
                Y.append('Neither')
            if row[category_zero] == 0 and row[category_one] == 1:
                Y.append(category_one)
            if row[category_zero] == 1 and row[category_one] == 0:
                Y.append(category_zero)
            if row[category_zero] == 1 and row[category_one] == 1:
                Y.append('Both')

        if index % 10000 == 0:
            print('Done ' + str(index) + ' of ' + str(total))

    u.show_2d_pca(X, Y, ['red', 'green', 'blue', 'purple'])

def show_disaster_pca_avgvec(disaster_df, category_name):
    """
    Show 2D pca for given category based on disaster data
    :param disaster_df: The NORMALIZED disaster df
    :param category_name: The category for PCA
    """

    word_frequency = get_disaster_word_frequency(disaster_df)

    total = u.row_count(disaster_df)

    words = []
    X = []
    Y = []
    for index, row in disaster_df.iterrows():

        message_vectors = []
        for word in row['message']:
            # Disregards less than 50 instances
            if word_frequency[word] < 50:
                continue

            # Process the rest
            vector, op_success = u.try_word2vec(word)
            if op_success:
                words.append(word)
                message_vectors.append(vector)

        if message_vectors:
            X.append(np.average(message_vectors, axis=0))
            Y.append(row[category_name])

        if index%10000==0:
            print('Done ' + str(index) + ' of ' + str(total))

    u.show_2d_pca(X, Y, ['r', 'g'])

def try_nn_avgvec_with(disaster_df, category_name, outout_model_filename):
    """
    Try training a simple NN to predict the given category (Averages word vectors in 1 message)
    :param disaster_df: The NORMALIZED disaster df
    :param category_name: The output category name
    :param outout_model_filename: The file path to output the model to
    """

    word_frequency = get_disaster_word_frequency(disaster_df)

    total = u.row_count(disaster_df)

    words = []
    X = []
    Y = []
    for index, row in disaster_df.iterrows():

        message_vectors = []
        for word in row['message']:
            # Disregards less than 50 instances
            if word_frequency[word] < 50:
                continue

            # Process the rest
            vector, op_success = u.try_word2vec(word)
            if op_success:
                words.append(word)
                message_vectors.append(vector)

        if message_vectors:
            X.append(np.average(message_vectors, axis=0))
            Y.append(row[category_name])

        if index%10000==0:
            print('Done ' + str(index) + ' of ' + str(total))

    nn_train_save_show_results(
        X,
        Y,
        hidden_layer_sizes=(8, 5, 5, 5),
        model_file_name=outout_model_filename,
        solver='lbfgs',
        max_iter=100000)

def try_nn_with(disaster_df, category_name):
    """
    Try training a neural network for the given category output
    :param disaster_df: The NORMALIZED disaster df
    :param category_name: The category name
    """

    word_frequency = get_disaster_word_frequency(disaster_df)

    total = u.row_count(disaster_df)

    words = []
    X = []
    Y = []
    for index, row in disaster_df.iterrows():

        for word in row['message']:
            # Dis-regard less than 50 instances
            if word_frequency[word] < 50:
                continue

            # Process the rest
            words.append(word)
            vector, op_success = u.try_word2vec(word)
            if(op_success):
                X.append(vector)
                Y.append(row[category_name])

        if index%10000==0:
            print('Done ' + str(index) + ' of ' + str(total))


    nn_train_save_show_results( X,
                                Y,
                                hidden_layer_sizes=(60, 30),
                                model_file_name='investigation_results/try_nn/first_model.pkl')

def nn_train_save_show_results(X, Y, hidden_layer_sizes, model_file_name, test_size=0.3, solver='sgd', alpha=1e-5, random_state=1, max_iter=200):
    """
    Given X, Y, train an NN, save it, and show the confusion matrix
    :param X: The input
    :param Y: The output
    :param hidden_layer_sizes: The hidden layer configuration
    :param model_file_name: The name of the file save the trained model to
    :param test_size: (MLPClassifier param) The fraction of data to use for test
    :param solver: (MLPClassifier param) The solver to use
    :param alpha: (MLPClassifier param) Alpha
    :param random_state: (MLPClassifier param) Random state
    :param max_iter: (MLPClassifer param) The maximum iterations to perform
    """

    # Train the model
    model = nn.MLPClassifier(
        solver=solver,
        alpha=alpha,
        hidden_layer_sizes=hidden_layer_sizes,
        random_state=random_state,
        max_iter=max_iter)

    x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size=test_size, random_state=random_state)
    model.fit(x_train, y_train)

    # Save the model
    u.to_pkl(model, model_file_name)

    # Assess the model
    y_pred = model.predict(x_test)
    matrix = me.confusion_matrix(y_test, y_pred)
    pd.DataFrame(matrix).to_csv('confusion_matrix.csv',index=False)
    print(matrix)
    sns.heatmap(pd.DataFrame(matrix))
    plt.show()

def create_normalized_disaster_to(file_name):
    """
    Normalizes disaster and creates a csv file with the resulting data
    :param file_name: The name to output to
    """

    disaster = u.read_csv('../data/disaster.csv')\
        .pipe(nlp.remove_columns)\
        .pipe(nlp.one_hot_encode_genre)\
        .pipe(nlp.normalize_related_category_values)\
        .pipe(nlp.normalize_messages)

    disaster.to_csv(file_name, index=False)

def get_disaster_word_frequency(disaster_df):
    """
    Get the word frequency for the given disaster_df (Must have a 'message' column)
    :return: A word frequency mapping
    """

    word_count = {}
    for index, row in disaster_df.iterrows():
        for word in row['message']:
            word_count[word] = 1 if word not in word_count else word_count[word] + 1

    return word_count

def show_weather_pca(word_vector_dir, weather_words_csv):
    """
    Given names of the csv files, plots PCA for weather words vs normal word sample
    :param word_vector_dir: The file name of the csv containing the sample normal words
    :param weather_words_csv: The file name of the csv containing weather related words
    :return:
    """

    weather_words = u.read_csv(weather_words_csv)

    X = []
    Y = []
    for filename in os.listdir(word_vector_dir):

        X.append(np.load(word_vector_dir + '/' + filename))

        if weather_words['word'].str.contains(filename.replace('.npy', '')).any():
            Y.append('weather_related')
        else:
            Y.append('general')

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(X)
    finalDf= pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    finalDf['Category'] = pd.Series(Y)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = ['weather_related', 'general']
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['Category'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=5)
    ax.legend(targets)
    ax.grid()

    plt.show()

def create_word_vectors(model_bin_file, weather_words_csv, all_words_csv, output_dir, all_word_sample_size=500):

    weather = u.read_csv(weather_words_csv)
    all = u.read_csv(all_words_csv).sample(all_word_sample_size)

    model = gensim.models.KeyedVectors.load_word2vec_format(model_bin_file, binary=True)

    for index, row in weather.iterrows():
        try:
            np.save(output_dir + '/' + row['word'], model[row['word']])
        except:
            pass

    for index, row in all.iterrows():
        try:
            np.save(output_dir + '/' + row['word'], model[row['word']])
        except:
            pass

def print_disaster_category_values():
    """
    Prints all the disaster category values (To find out if the '2's are a mistake)
    """
    disaster = u.read_csv('data/disaster.csv')
    non_cat_names = ['id', 'message', 'original', 'genre']

    for cat in list(dropwhile(lambda x: x in non_cat_names, disaster.columns)):
        print(cat)
        print('-------------------------')
        for value in disaster[cat].unique():
            print(str(value) + ' - ' + str(u.row_count(disaster[disaster[cat]==value])))
        print()

def create_readble_bias(bias_file_name, database_filename, table_name):
    """
    Based on the bias file output, creates new table and saves it to an SQLite DB
    :param bias_file_name: The file with all the word ==> category indicator data
    :param database_filename: The database file name
    :param table_name: The name of the table
    """

    bias = u.read_csv(bias_file_name)
    readable_bias = pd.DataFrame()

    for column in list(dropwhile(lambda x: '_bias' not in x, bias.columns)):

        category = column.replace('_bias', '')

        bias = bias.sort_values(by=[column], ascending=False)

        readable_bias[category + '_word'] = bias['word']
        readable_bias[category + '_ones'] = bias[category + '_ones']
        readable_bias[category + '_total'] = bias[category + '_total']
        readable_bias[category + '_bias'] = bias[category + '_bias']


    u.save_df_to_sqlite(readable_bias, database_filename, table_name)

def create_word_bias_data(disaster_csv, bias_file_name):
    """
    Based on the disaster data, generates a file to store the bias data for word ==> category
    :param disaster_csv: The disaster.csv file path
    :param bias_file_name: The file name of the output file with bias data
    """

    # Read data
    disaster = u.read_csv(disaster_csv)
    disaster['message'] = disaster['message'].apply(ast.literal_eval)
    non_category_names = ['id', 'message', 'original', 'genre_direct', 'genre_news', 'genre_social']
    category_names = list(dropwhile(lambda x: x in non_category_names, disaster.columns))

    # Record word to category frequency mapping
    bias_data = {}
    total = u.row_count(disaster)
    for index, row in disaster.iterrows():

        for word in row['message']:

            if word not in bias_data:
                bias_data[word] = {}
                for category_name in category_names:
                    bias_data[word][category_name + '_ones'] = 0
                    bias_data[word][category_name + '_total'] = 0

            for category_name in category_names:
                bias_data[word][category_name + '_ones'] += row[category_name]
                bias_data[word][category_name + '_total'] += 1

        if index%100==0:
            print('Done ' + str(index) + ' of ' + str(total))

    # Generate a data frame from the frequency mapping
    bias = pd.DataFrame()
    bias['word'] = bias_data.keys()

    # Populate each category ones and total column and add it to dataframe
    columns = bias_data[next(iter(bias_data))].keys()
    current_column_data = []
    i = 1
    for column in columns:
        for word in bias_data:
            current_column_data.append(bias_data[word][column])

        bias[column] = current_column_data
        current_column_data = []
        i += 1

    # For each category, calculate the bias based on the ones and total data
    for category_name in category_names:
        bias[category_name + '_bias'] = bias[category_name + '_ones'] / bias[category_name + '_total']

    bias.to_csv(bias_file_name, index=False)

def find_most_biased_word_for(category_name):
    """
    Goes into the disaster.csv and prints the words that are the strongest indicator of the given category
    :param category_name: The name fo the target category
    """

    disaster = u.read_csv('disaster.csv')

    num_rows = u.row_count(disaster)

    word_target_count = {}
    for index, row in disaster.iterrows():

        for word in row['message'].upper().split(' '):

            if word not in word_target_count:
                word_target_count[word] = [0, 0, 0]

            word_target_count[word][row[category_name]] += 1
            word_target_count[word][2] = word_target_count[word][1] / word_target_count[word][0] if word_target_count[word][0] > 0 else 2147483648

        if index%5000==0:
            print('Done ' + str(index) + ' of ' + str(num_rows))

    word_corrs = pd.DataFrame()
    word_corrs['word'] = word_target_count.keys()
    word_corrs['zeros'] = pd.Series(map(lambda x: x[0], word_target_count.values()))
    word_corrs['ones'] = pd.Series(map(lambda x: x[1], word_target_count.values()))
    word_corrs['one2zero'] = pd.Series(map(lambda x: x[2], word_target_count.values()))

    word_corrs = word_corrs.sort_values(by=['one2zero'], ascending=False)
    word_corrs.to_csv('word_corrs.csv', index=False)

    for index, row in word_corrs[word_corrs['one2zero'] < 2147483648].iterrows():

        print(row['word'] + ' - Ones: ' + str(row['ones']) + ', Zeros: ' + str(row['zeros']))
        input()

def show_disaster_pca_for(category_name):
    """
    Show a PCA where the data points are the word vectors and the targets are the values in the given category
    :param category_name: The disaster category name
    """

    model = gensim.models.Word2Vec.load('disaster.model')

    disaster = u.read_csv('disaster.csv')

    X = []
    Y = []

    num_rows = u.row_count(disaster)

    for index, row in disaster.iterrows():
        for word in row['message'].upper().split(' '):
            if word in model.wv.vocab:
                X.append(model[word])
                Y.append(row[category_name])

        if index %5000==0:
            print('Done ' + str(index) + ' of ' + str(num_rows) + ' rows')

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(X)
    finalDf= pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    finalDf['Is' + category_name] = pd.Series(Y)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = [0, 1]
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['Is' + category_name] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=5)
    ax.legend(targets)
    ax.grid()

    plt.show()

def print_word_frequency():
    """
    Prints the word frequency in messages, from most frequent word to least frequent
    """

    messages = u.read_csv('../disaster.csv')

    message_words = messages['message'].apply(lambda x: x.lower().split(' '))

    word_count = {}
    for message in message_words:

        for word in message:

            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    for key, value in sorted(word_count.items(), key=lambda item: item[1], reverse=True):
        print(key + ' - ' + str(value))

def print_disaster_dupe_summary():
    """
    Goes through merged, and categorized disaster.csv and prints the ids that are duplicates and a preview of the
    messages
    """

    disaster = u.read_csv('../data/disaster.csv')

    # Check for dupes
    ids = set()
    disaster['id'].apply(lambda x: ids.add(x))

    dupe_ids = []
    for id in ids:
        if u.row_count(disaster[disaster['id'] == id]) > 1:
            print(id)
            dupe_ids.append(id)

    for dupe_id in dupe_ids:
        print(disaster[disaster['id'] == dupe_id]['message'])

def print_unique_lengths_of_categories():
    """
    Prints all the different lengths that the 'categories' column has
    (If this print more than 1 number, than the data has a problem)
    """

    lengths = set()
    categories = u.read_csv('../data/disaster_categories.csv')

    for index, row in categories.iterrows():

        lengths.add(len(row['categories'].split(';')))


    for length in lengths:
        print(length)