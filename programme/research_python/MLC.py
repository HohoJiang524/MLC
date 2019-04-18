# -*- coding: utf-8 -*-

# libraries
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import arff


# read arff file
# return a dictionary, containing contents of the file.
def read_arff(arff_file, numLabels):
        obj = arff.load(open(arff_file, 'r'))
        data = np.array(obj['data']).astype('float')
        attributes = data[:, 0:data.shape[1] - numLabels]
        labels = data[:, data.shape[1] - numLabels:data.shape[1]]
        return {'attributes': attributes,
                'labels': labels,
                'names': obj['attributes']
                }


def naiveBayes_train(X, y):
        clf = GaussianNB()
        return clf.fit(X, y)


def naiveBayes_predict(X, model):
        return model.predict(X)


def lead(X, y, method='NaiveBayes'):
        if method == 'NaiveBayes':
                # Step1: train q classifiers using method
                error_array = np.zeros((y.shape))
                for i in range(y.shape[1]):
                        y_i = y[:, i]
                        model = naiveBayes_train(X, y_i)
                        y_pred = naiveBayes_predict(X, model)
                        error = y_i - y_pred
                        error_array[:, i] = error

                return error_array

                # Step2: use error


# __main__():
file = r'/Users/jiangjunhao/Desktop/ESKDB_HDP/data/small_datasets/emotions/emotions.arff'
data = read_arff(file,6)

X = data['attributes']
y = data['labels']
error_arr = lead(X,y)

for i in range(error_arr.shape[0]):
    print(error_arr[i,:])

