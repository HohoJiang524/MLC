# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
import random
import os
import sys
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from evaluation_metrics import evaluation
from skmultilearn.model_selection import iterative_train_test_split

def build_BN(labelFile, labelName, savePng):
    cmd = """cd ../programme/Chordalysis/ 
    java -Xmx1g -classpath bin:lib/core/commons-math3-3.2.jar:lib/core/jayes.jar:lib/core/jgrapht-jdk1.6.jar:lib/extra/jgraphx.jar:lib/loader/weka.jar demo.Run %s 0.05 %s false
    """ % (labelFile,savePng)

    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    out,err = p.communicate()
    for line in out.splitlines():
        if line.decode("utf-8").startswith('['):
            graph_set = [i for i in map(lambda x: x.split(','), line.decode("utf-8").replace(' ',',').strip('[[\,]]').split(',]['))]

    dic = {}
    for l in labelName:
        s = set()
        for i in map(lambda x: set(x) if l in x else None, graph_set):
            if i != None:
                s.update(i)
        s.remove(l)
        dic[l] = s

    return dic

# BR for getting error matrix
def naiveBayes_multi_label_training_BR(X_train, y_train):
    start = time.time()

    n_label = y_train.shape[1]
    classifier_list = [MultinomialNB() for i in range(n_label)]
    for i in range(n_label):
        classifier_list[i].fit(X_train, y_train.iloc[:, i])

    end = time.time()
    training_time = end - start

    return classifier_list, training_time


def naiveBayes_multi_label_testing_BR(X_test, n_label, classifier_list):
    y_predict = pd.DataFrame()
    y_prob = pd.DataFrame()

    start = time.time()

    for i in range(n_label):
        y_predict_i = classifier_list[i].predict(X_test)
        y_predict = pd.concat([y_predict, pd.DataFrame(y_predict_i)], axis=1)

        y_predict_prob_i = classifier_list[i].predict_proba(X_test)[:, 1]
        y_prob = pd.concat([y_prob, pd.DataFrame(y_predict_prob_i)], axis=1)

    end = time.time()
    testing_time = end - start

    return y_predict, y_prob, testing_time


def BR_test(data, label, dataPath, random_state=3071980):
    # data set information
    n_label = label.shape[1]
    # split training and test data set
    X_train, y_train, X_test, y_test = iterative_train_test_split(np.matrix(data), np.matrix(label), test_size=0.5)

    X_train = pd.DataFrame(X_train, columns=data.columns)
    X_test = pd.DataFrame(X_test, columns=data.columns)

    y_train = pd.DataFrame(y_train, columns=label.columns)
    y_test = pd.DataFrame(y_test, columns=label.columns)


    # training
    classifier_list, training_time = naiveBayes_multi_label_training_BR(X_train, y_train)

    # testing
    y_predict, y_prob, testing_time = naiveBayes_multi_label_testing_BR(X_test, n_label, classifier_list)

    y_predict.columns = label.columns
    return y_predict, y_test


def naiveBayes_multi_label_training_order(X_train, y_train, bayes_net, order):
    start = time.time()

    n_label = y_train.shape[1]

    classifier_list = [MultinomialNB() for i in range(n_label)]  # create a classifier chain

    learned_label = []

    for i in range(n_label):
        if i == 0:
            l = order[i]
            classifier_list[i].fit(X_train, y_train.loc[:, l])
            learned_label.append(l)

        else:
            l = order[i]
            par = [x for x in bayes_net[l] if x in learned_label]
            X = pd.concat([X_train, y_train.loc[:, par]], axis=1)  # put the previous label into attribute space
            classifier_list[i].fit(X, y_train.loc[:, l])
            learned_label.append(l)

    end = time.time()
    training_time = end - start

    return classifier_list, learned_label


def naiveBayes_multi_label_testing_order(X_test, n_label, classifier_list, bayes_net, learned_label):
    y_predict = pd.DataFrame(index=X_test.index)
    y_prob = pd.DataFrame(index=X_test.index)
    y_true = pd.DataFrame(index=X_test.index)

    start = time.time()

    predicted_list = []

    for i in range(n_label):
        if i == 0:
            l = learned_label[i]
            y_predict_i = classifier_list[i].predict(X_test)
            y_predict_prob_i = classifier_list[i].predict_proba(X_test)[:, 1]
            y_predict = pd.concat([y_predict, pd.DataFrame(y_predict_i, index=X_test.index, columns=[l])], axis=1)
            y_prob = pd.concat([y_prob, pd.DataFrame(y_predict_prob_i, index=X_test.index, columns=[l])], axis=1)
            predicted_list.append(l)

        else:
            l = learned_label[i]
            par = [p for p in bayes_net[l] if p in predicted_list]
            if len(par) != 0:
                X = pd.concat([X_test, y_predict.loc[:, par]], axis=1)  # put the previous label into attribute space
            else:
                X = X_test
            y_predict_i = classifier_list[i].predict(X)
            y_predict_prob_i = classifier_list[i].predict_proba(X)[:, 1]

            y_predict = pd.concat([y_predict, pd.DataFrame(y_predict_i, index=X_test.index, columns=[l])], axis=1)
            y_prob = pd.concat([y_prob, pd.DataFrame(y_predict_prob_i, index=X_test.index, columns=[l])], axis=1)

            predicted_list.append(l)

    return y_predict, y_prob


def BCC_test_order(data, label, dataPath, bayes_net, random_state=3071980, ensemble=5, order_method="random"):
    # data set information
    n_label = label.shape[1]
    n_attr = data.shape[1]
    n_instance = data.shape[0]
    avg_label_per_instance = label.sum(axis=1).mean()

    # get order
    if order_method == "best_prediction":
        y_predict, y_test = BR_test(data, label, dataPath, 3071980)
        acc = (y_predict.values == y_test.values).mean(axis=0)
        order = list(label.columns[np.argsort(-acc)])

    elif order_method == "largest_edges":
        a = [(x, len(y)) for x, y in bayes_net.items()]
        a_sort = sorted(a, key=lambda x: x[1], reverse=True)
        order = [x[0] for x in a_sort]

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
        if order_method == "random":
            order = random.sample(list(range(n_label)), n_label)  # get orders

        # training
        # print("--- start training ---\n")
        classifier_list, learned_label = naiveBayes_multi_label_training_order(X_train, y_train, bayes_net, order)

        # testing
        # print("--- start testing ---\n")
        y_predict, y_prob = naiveBayes_multi_label_testing_order(X_test, n_label, classifier_list, bayes_net,
                                                                 learned_label)

        y_predict = y_predict[label.columns]
        y_prob = y_prob[label.columns]

        y_pred_ensemble = y_pred_ensemble + y_predict
        y_prob_ensemble = y_prob_ensemble + y_prob

    y_pred_ensemble = (((y_pred_ensemble / ensemble) >= 0.5) * 1).astype('int')
    y_prob_ensemble = y_prob_ensemble / ensemble
    y_pred_ensemble = y_pred_ensemble.fillna(0)
    y_prob_ensemble = y_prob_ensemble.fillna(0)

    # evaluation
    performance = evaluation(y_pred_ensemble, y_prob_ensemble, y_test)

    performance_df = pd.DataFrame.from_dict(performance, orient='index')

    return performance_df


def BCC_test_order_twofold(data, label, dataPath, bayes_net, random_state=3071980, ensemble=5, order_method="random"):
    # data set information
    n_label = label.shape[1]
    n_attr = data.shape[1]
    n_instance = data.shape[0]
    avg_label_per_instance = label.sum(axis=1).mean()

    # get order
    if order_method == "best_prediction":
        y_predict, y_test = BR_test(data, label, dataPath, 3071980)
        acc = (y_predict.values == y_test.values).mean(axis=0)
        order = list(label.columns[np.argsort(-acc)])

    elif order_method == "largest_edges":
        a = [(x, len(y)) for x, y in bayes_net.items()]
        a_sort = sorted(a, key=lambda x: x[1], reverse=True)
        order = [x[0] for x in a_sort]

    # split training and test data set
    X_train, y_train, X_test, y_test = iterative_train_test_split(np.matrix(data), np.matrix(label), test_size=0.5)

    X_train = pd.DataFrame(X_train, columns=data.columns)
    X_test = pd.DataFrame(X_test, columns=data.columns)

    y_train = pd.DataFrame(y_train, columns=label.columns)
    y_test = pd.DataFrame(y_test, columns=label.columns)

    performance_df_all = pd.DataFrame()
    for j in range(2):
        X_test, X_train = X_train, X_test
        y_test, y_train = y_train, y_test
        # ensemble
        y_pred_ensemble = pd.DataFrame(np.zeros(y_test.shape), columns=y_test.columns, index=y_test.index)
        y_prob_ensemble = pd.DataFrame(np.zeros(y_test.shape), columns=y_test.columns, index=y_test.index)

        for i in range(ensemble):
            if order_method == "random":
                order = random.sample(list(range(n_label)), n_label)  # get orders

            # training
            # print("--- start training ---\n")
            classifier_list, learned_label = naiveBayes_multi_label_training_order(X_train, y_train, bayes_net, order)

            # testing
            # print("--- start testing ---\n")
            y_predict, y_prob = naiveBayes_multi_label_testing_order(X_test, n_label, classifier_list, bayes_net,
                                                                     learned_label)

            y_predict = y_predict[label.columns]
            y_prob = y_prob[label.columns]

            y_pred_ensemble = y_pred_ensemble + y_predict
            y_prob_ensemble = y_prob_ensemble + y_prob

        y_pred_ensemble = (((y_pred_ensemble / ensemble) >= 0.5) * 1).astype('int')
        y_prob_ensemble = y_prob_ensemble / ensemble
        y_pred_ensemble = y_pred_ensemble.fillna(0)
        y_prob_ensemble = y_prob_ensemble.fillna(0)

        # evaluation
        performance = evaluation(y_pred_ensemble, y_prob_ensemble, y_test)
        performance_df = pd.DataFrame.from_dict(performance, orient='index')
        performance_df_all = pd.concat([performance_df_all, performance_df],axis=1)

    return performance_df_all

if __name__ == "__main__":
    # get dataset from the command line input
    #dataset = "yeast"
    dataset = sys.argv[1]
    ensemble = sys.argv[2]

    # setup
    savePath = "../result/order/best_prediction_NB_BCC/"
    dataPath = '../data/' + dataset + "/"
    X_file = "X_scale.csv"
    y_file = "y.csv"

    # read data
    data = pd.read_csv(os.path.join(dataPath, X_file))
    label = pd.read_csv(os.path.join(dataPath, y_file))

    labelFile = os.path.abspath(dataPath)+"/y.csv"
    savePng = os.path.abspath("./temp/bayes_net.png")

    df_result = pd.DataFrame()

    seed = [21313,34132,43413,62423,56576]
    for s in seed:
        bayes_net = build_BN(labelFile, label.columns, savePng)
        if ensemble == "T":
            df = BCC_test_order_twofold(data, label, dataPath, bayes_net, s, 10, order_method="best_prediction")
        else:
            df = BCC_test_order_twofold(data, label, dataPath, bayes_net, s, 1, order_method="best_prediction")

        df_result = pd.concat([df_result, df], axis=1)

    df_result.columns = range(df_result.shape[1])

    # save the results
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    df_result.to_csv(savePath+dataset+".csv")