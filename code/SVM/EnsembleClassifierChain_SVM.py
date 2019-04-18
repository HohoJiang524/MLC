# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
import random
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn import svm
from evaluation_metrics import evaluation
from skmultilearn.model_selection import iterative_train_test_split


def naiveBayes_multi_label_training(X_train, y_train):
    start = time.time()

    n_label = y_train.shape[1]

    order = random.sample(list(range(n_label)), n_label)  # get orders

    classifier_list = [svm.SVC(gamma='auto', probability=True) for i in range(n_label)]  # create a classifier chain

    for i in range(n_label):
        if i == 0:
            if y_train.iloc[:, order[i]].nunique() == 1:
                classifier_list[i] = None
            else:
                classifier_list[i].fit(X_train, y_train.iloc[:, order[i]])

        else:
            if y_train.iloc[:, order[i]].nunique() == 1:
                X_train = pd.concat([X_train, y_train.iloc[:, order[i - 1]]],
                                    axis=1)  # put the previous label into attribute space
                classifier_list[i] = None
            else:
                X_train = pd.concat([X_train, y_train.iloc[:, order[i - 1]]],
                                    axis=1)  # put the previous label into attribute space
                classifier_list[i].fit(X_train, y_train.iloc[:, order[i]])

    end = time.time()
    training_time = end - start

    return classifier_list, training_time, order


def naiveBayes_multi_label_testing(X_test, n_label, classifier_list, order):
    y_predict = pd.DataFrame(index=X_test.index)
    y_prob = pd.DataFrame(index=X_test.index)
    y_true = pd.DataFrame(index=X_test.index)

    start = time.time()

    for i in range(n_label):
        if classifier_list[i] == None:
            y_predict_i = np.array([0] * X_test.shape[0])
            y_predict_prob_i = np.array([0] * X_test.shape[0])
        else:
            y_predict_i = classifier_list[i].predict(X_test)
            y_predict_prob_i = classifier_list[i].predict_proba(X_test)[:, 1]

        y_predict = pd.concat([y_predict, pd.DataFrame(y_predict_i, index=X_test.index)], axis=1)

        y_prob = pd.concat([y_prob, pd.DataFrame(y_predict_prob_i, index=X_test.index)], axis=1)

        X_test = pd.concat([X_test, pd.DataFrame(y_predict_i, index=X_test.index)], axis=1,
                           ignore_index=True)  # put the previous label into attribute space
    end = time.time()
    testing_time = end - start

    return y_predict, y_prob, testing_time


def ECC_test(data, label, random_state=3071980, ensemble=5):
    # data set information
    n_label = label.shape[1]

    # split training and test data set
    X_train, y_train, X_test, y_test = iterative_train_test_split(np.matrix(data), np.matrix(label), test_size=0.5)
    X_train = pd.DataFrame(X_train, columns=data.columns)
    X_test = pd.DataFrame(X_test, columns=data.columns)
    y_train = pd.DataFrame(y_train, columns=label.columns)
    y_test = pd.DataFrame(y_test, columns=label.columns)

    # ensemble
    y_pred_ensemble = pd.DataFrame(np.zeros(y_test.shape), columns=y_test.columns, index=y_test.index)
    y_prob_ensemble = pd.DataFrame(np.zeros(y_test.shape), columns=y_test.columns, index=y_test.index)
    for i in range(ensemble):
        # training
        # print("--- start training ---\n")
        classifier_list, training_time, order = naiveBayes_multi_label_training(X_train, y_train)

        # testing
        # print("--- start testing ---\n")
        y_predict, y_prob, testing_time = naiveBayes_multi_label_testing(X_test, n_label, classifier_list, order)

        y_predict.columns = label.columns[order]
        y_prob.columns = label.columns[order]
        y_predict = y_predict[label.columns]
        y_prob = y_prob[label.columns]

        y_pred_ensemble = y_pred_ensemble + y_predict
        y_prob_ensemble = y_prob_ensemble + y_prob

    y_pred_ensemble = (((y_pred_ensemble / ensemble) >= 0.5) * 1).astype('int')
    y_prob_ensemble = y_prob_ensemble / ensemble

    # evaluation
    performance = evaluation(y_pred_ensemble, y_prob_ensemble, y_test)

    performance_df = pd.DataFrame.from_dict(performance, orient='index')

    return performance_df


def ECC_test_2_fold(data, label, random_state=3071980, ensemble=5):
    # data set information
    n_label = label.shape[1]

    # split training and test data set
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.5, random_state=random_state)

    performance_df_all = pd.DataFrame()

    for j in range(2):
        X_test, X_train = X_train, X_test
        y_test, y_train = y_train, y_test

        # ensemble
        y_pred_ensemble = pd.DataFrame(np.zeros(y_test.shape), columns=y_test.columns, index=y_test.index)
        y_prob_ensemble = pd.DataFrame(np.zeros(y_test.shape), columns=y_test.columns, index=y_test.index)
        for i in range(ensemble):
            # training
            # print("--- start training ---\n")
            classifier_list, training_time, order = naiveBayes_multi_label_training(X_train, y_train)

            # testing
            # print("--- start testing ---\n")
            y_predict, y_prob, testing_time = naiveBayes_multi_label_testing(X_test, n_label, classifier_list, order)

            y_predict.columns = label.columns[order]
            y_prob.columns = label.columns[order]
            y_predict = y_predict[label.columns]
            y_prob = y_prob[label.columns]

            y_pred_ensemble = y_pred_ensemble + y_predict
            y_prob_ensemble = y_prob_ensemble + y_prob

        y_pred_ensemble = (((y_pred_ensemble / ensemble) >= 0.5) * 1).astype('int')
        y_prob_ensemble = y_prob_ensemble / ensemble

        # evaluation
        performance = evaluation(y_pred_ensemble, y_prob_ensemble, y_test)
        performance_df = pd.DataFrame.from_dict(performance, orient='index')

        #performance_df_all.index = performance_df.index

        performance_df_all = pd.concat([performance_df_all, performance_df],axis=1)

    return performance_df_all

if __name__ == "__main__":
    # get dataset from the command line input
    #dataset = "yeast"
    dataset = sys.argv[1]

    # setup
    savePath = "../result/SVM/ECC_SVM/"
    dataPath = '../data/' + dataset + "/"
    X_file = "X_scale.csv"
    y_file = "y.csv"

    # read data
    data = pd.read_csv(os.path.join(dataPath, X_file))

    label = pd.read_csv(os.path.join(dataPath, y_file))

    # run algorithm with 5 times 2-fold cross validation.
    df_result = pd.DataFrame()  # dataframe to store 10 results
    seed = [21313,34132,43413,62423,56576]
    for s in seed:
        df = ECC_test_2_fold(data, label, random_state=s, ensemble=10)
        df_result = pd.concat([df_result,df],axis=1)
    df_result.columns = range(df_result.shape[1])

    # save the results
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    df_result.to_csv(savePath+dataset+".csv")