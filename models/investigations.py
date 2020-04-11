import gensim
import string

import utility.util as u
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn import svm
from itertools import *

def create_disaster_category_values_hist():
    """
    Creates a histogram of all the disaster category values (To find out if the '2's are a mistake)
    """

    disaster = u.read_csv('')

def create_readble_bias(bias_file_name, best_words_file):
    """
    Based on the bias file output, shows the best indicator words for each category in the output file
    :param bias_file_name: The file with all the word ==> category indicator data
    :param best_words_file: The file name of the output file with the best words for each category
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

    readable_bias.to_csv(best_words_file, index=False)




def create_word_bias_data(disaster_csv, bias_file_name):
    """
    Based on the disaster data, generates a file to store the bias data for word ==> category
    :param disaster_csv: The disaster.csv file path
    :param bias_file_name: The file name of the output file with bias data
    """

    # Read data
    disaster = u.read_csv(disaster_csv)
    non_category_names = ['id', 'message', 'original', 'genre']
    category_names = list(dropwhile(lambda x: x in non_category_names, disaster.columns))

    # Record word to category frequency mapping
    bias_data = {}
    total = u.row_count(disaster)
    for index, row in disaster.iterrows():

        for word in row['message'].upper().split(' '):

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

def say_hello():

    print('Hello World')