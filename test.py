"""
This script is supposed to take the test data stored into a .pickle file generated from sample.py,
the trained model generated from train.py, and calculate the recall, precision, accuracy and
f1 scores of the model. 

The script takes three arguments in the command line:
- the file containing the test data (test.pickle)
- the file with the trained model (output.pickle)
- the desired type for the average score(micro or macro)

"""



import argparse
import pickle
import numpy as np
import sklearn.metrics as skm
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



def open_file(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


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
    
    test_X = np.array(counter)
    test_y = np.array(consonant_list)

    return test_X, test_y

def eval_model(model, test_X, test_y, average):
    y_pred = model.predict(test_X)
    prec_score = precision_score(test_y, y_pred, average=average)
    rec_score = recall_score(test_y, y_pred, average=average)
    F1_score = f1_score(test_y, y_pred, average=average)
    accuracy = skm.accuracy_score(test_y, y_pred)


    print("The accuracy score is:", accuracy)
    print("The precision score is:", prec_score)
    print("The recall score is:", rec_score)
    print("The f1 score is:", F1_score)    

if __name__ =="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", help = "The test file obtained from sample.py")
    parser.add_argument("model_file", help = "The trained model file obtained from train.py")
    parser.add_argument("average", help = "Micro or macro average")
    
    args = parser.parse_args()
    
    test_data = open_file(args.test_file)
    trained_model = open_file(args.model_file)

    test_X, test_y = create_df(test_data)
    scores = eval_model(trained_model, test_X, test_y, args.average)
    print(scores)