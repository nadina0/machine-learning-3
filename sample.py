"""
This script takes randomised data from a corpus of sentences, which is used
for a consonant predictor, and generates samples which will be used to train
a model. The samples (tuples of four letters and the consonant after them) 
are split into training data and testing data.

The script takes five arguments in the command line:
-the corpus file (filename.gz)
-the number of samples to generate
-the ratio of splitting the data (the threshold, for example, 20)
-the file in which the training data will go (train.pickle)
-the file in which the testing data will go (test.pickle)

"""

import argparse
import gzip
import random
import pickle

def sample_lines(file, number):
    zipped_file = gzip.open(file, 'rb')
    all_lines = zipped_file.readlines()
    lines_list = []
    for line in all_lines:
        line_decode = line.decode('utf8').strip().lower()
        lines_list.append(line_decode)
    return random.sample(lines_list, int(number))


consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']

def find_consonant(word):
    for character in word:
        if character in consonants:
            return character

def create_samples(sampled_lines):
    all_samples = []
    for sample in sampled_lines:
        for i in range(0, len(sample)-4):
            l1 = sample[i] + "_1"
            l2 = sample[i+1] + "_2"
            l3 = sample[i+2] + "_3"
            l4 = sample[i+3] + "_4"
            for letter in sample[i+4:]:
                if letter in consonants:
                    nearest_consonant = letter
                    break
                else:
                    continue
            first_letters = l1,l2,l3,l4
            all_samples.append((first_letters, nearest_consonant))
    return all_samples


def split(samples, size):
    ratio = round(len(samples) * (int(size)/100))
    train_set = samples[ratio:]
    test_set = samples[:ratio]

    return train_set, test_set

def convert_data(train_data, test_data, train_file, test_file):
    with open(train_file, 'wb')as file:
        pickle.dump(train_data, file)
    with open(test_file, 'wb') as file:
        pickle.dump(test_data, file)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="the corpus file used for obtaining the samples")
    parser.add_argument("sample_number", help="the desired number of samples")
    parser.add_argument("ratio", help="the percentage of test to train samples")
    parser.add_argument("test_file", help="the file with the test samples")
    parser.add_argument("train_file", help="the file with the train samples")

    args = parser.parse_args()

    sampled_lines = sample_lines(args.file, args.sample_number)
    samples = create_samples(sampled_lines)
    train_data, test_data = split(samples, args.ratio)
    convert_data(train_data, test_data, args.test_file, args.train_file)