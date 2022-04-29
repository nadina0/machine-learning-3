"""
This script takes the training data stored into a .pickle file generated from sample.py
and trains a model using it.

The script takes three arguments in the command line:
- the file containing the training samples (train.pickle)
- the desired model of choice (either SVC or NB)
- the file in which the output will be stored (output.pickle)

"""


import argparse
import numpy as np
import pickle
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB



def open_file(train_file):
    with open(train_file, 'rb') as file:
        train_data = pickle.load(file)
    return train_data


def create_df(all_samples):
    all_columns = []
    for tuple, consonant in all_samples:
        for letter in tuple:
            if letter not in all_columns:
                all_columns.append(letter)

    columns = [item for sublist in all_columns for item in sublist]
    all_columns.append("CONSONANT")

    consonant_list = []
    counter = []
    for tuples, consonants in all_samples:
        count = []
        for column in columns:
            if column in tuples:
                count.append(1)
            elif column not in tuples:
                count.append(0)
            else:
                continue
        counter.append(count)
        consonant_list.append(consonants)
    
    X = np.array(counter)
    y = np.array(consonant_list)

    return X, y


def train(X, y, model):
    if model == "SVC":
        clf = svm.SVC(kernel='linear', probability=True)

    elif model == "NB":
        clf = MultinomialNB()
    else:
        print("Choose between SVC or NB")

    final = clf.fit(X, y)
    return final

def save_model(model, trained_model):
    with open(trained_model, 'wb')as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="the file containing the training samples")
    parser.add_argument("model_type", help="choose between SVC and NB")
    parser.add_argument("output", help="the file with the model")

    args = parser.parse_args()

    all_samples = open_file(args.file)
    X, y = create_df(all_samples)
    model = train(X, y, args.model_type)
    save_model(model, args.output)
