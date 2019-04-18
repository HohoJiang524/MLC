# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
import random
import os
import sys
import subprocess
from sklearn.model_selection import train_test_split
from sklearn import svm
from evaluation_metrics import evaluation
from skmultilearn.model_selection import iterative_train_test_split

# BR for getting error matrix
def naiveBayes_multi_label_training_BR(X_train, y_train):
    start = time.time()

    n_label = y_train.shape[1]
    classifier_list = [svm.SVC(gamma='auto', probability=True) for i in range(n_label)]
    for i in range(n_label):
        if y_train.iloc[:, i].nunique() == 1:
            classifier_list[i] = None
        else:
            classifier_list[i].fit(X_train, y_train.iloc[:, i])

    end = time.time()
    training_time = end - start

    return classifier_list, training_time


def naiveBayes_multi_label_testing_BR(X_test, n_label, classifier_list):
    y_predict = pd.DataFrame()
    y_prob = pd.DataFrame()

    start = time.time()

    for i in range(n_label):
        if classifier_list[i] == None:
            y_predict_i = np.array([0] * X_test.shape[0])
            y_predict_prob_i = np.array([0] * X_test.shape[0])
        else:
            y_predict_i = classifier_list[i].predict(X_test)
            y_predict_prob_i = classifier_list[i].predict_proba(X_test)[:, 1]

        y_predict = pd.concat([y_predict, pd.DataFrame(y_predict_i)], axis=1)
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


def naiveBayes_multi_label_training(X_train, y_train, bayes_net, root_name):
    n_label = y_train.shape[1]

    classifier_list = [svm.SVC(gamma='auto', probability=True) for i in range(n_label)]  # create a classifier chain

    learned_label = []

    i = 0
    inde_node = 0

    for node, par in bayes_net.items():
        if par == set():
            l = node
            if y_train.loc[:, l].nunique() == 1:
                classifier_list[i] = None
            else:
                classifier_list[i].fit(X_train, y_train.loc[:, l])
            i += 1
            learned_label.append(l)
            inde_node += 1

    while True:
        if i == inde_node:
            l = root_name
            if y_train.loc[:, l].nunique() == 1:
                classifier_list[i] = None
            else:
                classifier_list[i].fit(X_train, y_train.loc[:, l])
            i += 1
            learned_label.append(l)
            children = bayes_net[l]

        else:
            children_sub = []
            for child in children:
                par = [p for p in bayes_net[child] if p in learned_label]
                X = pd.concat([X_train, y_train.loc[:, par]], axis=1)  # put the previous label into attribute space

                if y_train.loc[:, child].nunique() == 1:
                    classifier_list[i] = None
                else:
                    classifier_list[i].fit(X, y_train.loc[:, child])
                i += 1
                learned_label.append(child)
                children_sub.extend([p for p in bayes_net[child] if p not in learned_label])
            children = [p for p in set(children_sub) if p not in learned_label]

        if i >= n_label:
            break

    return classifier_list, learned_label


def naiveBayes_multi_label_testing(X_test, n_label, classifier_list, bayes_net, learned_label):
    y_predict = pd.DataFrame(index=X_test.index)
    y_prob = pd.DataFrame(index=X_test.index)
    y_true = pd.DataFrame(index=X_test.index)

    predicted_list = []
    i = 0

    inde_node = 0
    for node, par in bayes_net.items():
        if par == set():
            l = learned_label[i]
            if classifier_list[i]==None:
                y_predict_i = np.array([0] * X_test.shape[0])
                y_predict_prob_i = np.array([0] * X_test.shape[0])
            else:
                y_predict_i = classifier_list[i].predict(X_test)
                y_predict_prob_i = classifier_list[i].predict_proba(X_test)[:, 1]

            y_predict = pd.concat([y_predict, pd.DataFrame(y_predict_i, index=X_test.index, columns=[l])], axis=1)
            y_prob = pd.concat([y_prob, pd.DataFrame(y_predict_prob_i, index=X_test.index, columns=[l])], axis=1)

            predicted_list.append(l)

            i += 1
            inde_node += 1

    while True:
        if i == inde_node:
            l = learned_label[i]
            if classifier_list[i]==None:
                y_predict_i = np.array([0] * X_test.shape[0])
                y_predict_prob_i = np.array([0] * X_test.shape[0])
            else:

                y_predict_i = classifier_list[i].predict(X_test)
                y_predict_prob_i = classifier_list[i].predict_proba(X_test)[:, 1]

            y_predict = pd.concat([y_predict, pd.DataFrame(y_predict_i, index=X_test.index, columns=[l])], axis=1)
            y_prob = pd.concat([y_prob, pd.DataFrame(y_predict_prob_i, index=X_test.index, columns=[l])], axis=1)

            predicted_list.append(l)

            i += 1

        else:
            l = learned_label[i]
            par = [p for p in bayes_net[l] if p in predicted_list]

            if len(par) != 0:
                X = pd.concat([X_test, y_predict.loc[:, par]], axis=1)  # put the previous label into attribute space
            else:
                X = X_test
            if classifier_list[i]==None:
                y_predict_i = np.array([0] * X_test.shape[0])
                y_predict_prob_i = np.array([0] * X_test.shape[0])
            else:
                y_predict_i = classifier_list[i].predict(X)
                y_predict_prob_i = classifier_list[i].predict_proba(X)[:, 1]

            y_predict = pd.concat([y_predict, pd.DataFrame(y_predict_i, index=X_test.index, columns=[l])], axis=1)
            y_prob = pd.concat([y_prob, pd.DataFrame(y_predict_prob_i, index=X_test.index, columns=[l])], axis=1)

            i += 1
            predicted_list.append(l)

        if i >= n_label:
            break

    return y_predict, y_prob



def BCC_test(data, label, dataPath, bayes_net, random_state=3071980, ensemble=5, root=None):
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

    node_list = []
    for node, par in bayes_net.items():
        if par != set():
            node_list.append(node)

    en = 0
    for i in range(ensemble):
        if root != None:
            root_name = root
        else:
            root_name = label.columns[random.randint(0,label.shape[1]-1)]
            if root_name not in node_list:
                continue

            else:
                # training
                # print("--- start training ---\n")
                classifier_list, learned_label = naiveBayes_multi_label_training(X_train, y_train, bayes_net, root_name)

                # testing
                # print("--- start testing ---\n")
                y_predict, y_prob = naiveBayes_multi_label_testing(X_test, n_label, classifier_list, bayes_net,
                                                                   learned_label)

                y_predict = y_predict[label.columns]
                y_prob = y_prob[label.columns]

                y_pred_ensemble = y_pred_ensemble + y_predict
                y_prob_ensemble = y_prob_ensemble + y_prob

                en += 1

    y_pred_ensemble = (((y_pred_ensemble / en) >= 0.5) * 1).astype('int')
    y_prob_ensemble = y_prob_ensemble / en
    y_pred_ensemble = y_pred_ensemble.fillna(0)
    y_prob_ensemble = y_prob_ensemble.fillna(0)

    # evaluation
    performance = evaluation(y_pred_ensemble, y_prob_ensemble, y_test)

    performance_df = pd.DataFrame.from_dict(performance, orient='index')

    return performance_df


def BCC_test_2_fold(data, label, dataPath, bayes_net, random_state=3071980, ensemble=5, root=None):
    # data set information
    n_label = label.shape[1]
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

        node_list = []
        for node, par in bayes_net.items():
            if par != set():
                node_list.append(node)

        en = 0
        for i in range(ensemble):
            if root != None:
                root_name = root
            else:
                root_name = label.columns[random.randint(0,label.shape[1]-1)]
                if root_name not in node_list:
                    continue

                else:
                    # training
                    # print("--- start training ---\n")
                    classifier_list, learned_label = naiveBayes_multi_label_training(X_train, y_train, bayes_net,
                                                                                     root_name)

                    # testing
                    # print("--- start testing ---\n")
                    y_predict, y_prob = naiveBayes_multi_label_testing(X_test, n_label, classifier_list, bayes_net,
                                                                       learned_label)

                    y_predict = y_predict[label.columns]
                    y_prob = y_prob[label.columns]

                    y_pred_ensemble = y_pred_ensemble + y_predict
                    y_prob_ensemble = y_prob_ensemble + y_prob

                    en += 1

        y_pred_ensemble = (((y_pred_ensemble / en) >= 0.5) * 1).astype('int')
        y_prob_ensemble = y_prob_ensemble / en
        y_pred_ensemble = y_pred_ensemble.fillna(0)
        y_prob_ensemble = y_prob_ensemble.fillna(0)

        # evaluation
        performance = evaluation(y_pred_ensemble, y_prob_ensemble, y_test)
        performance_df = pd.DataFrame.from_dict(performance, orient='index')


        performance_df_all = pd.concat([performance_df_all, performance_df],axis=1)

    return performance_df_all


if __name__ == "__main__":
    # get dataset from the command line input
    dataset = sys.argv[1]
    ensemble = sys.argv[2]
    # setup
    savePath = "../result/SVM/LEAD_SVM/"
    dataPath = '../data/' + dataset + "/"
    X_file = "X_scale.csv"
    y_file = "y.csv"

    # read data
    data = pd.read_csv(os.path.join(dataPath, X_file))
    label = pd.read_csv(os.path.join(dataPath, y_file))

    savePng = os.path.abspath("./temp/bayes_net.png")

    df_result = pd.DataFrame()

    seed = [21313,34132,43413,62423,56576]
    for s in seed:
        y_predict, y_test = BR_test(data, label, dataPath, s)

        error_matrix = pd.DataFrame(np.array(y_predict) - np.array(y_test), columns=y_test.columns)

        labelFile = os.path.abspath(dataPath) + "/error_matrix.csv"
        error_matrix.to_csv(os.path.join(dataPath, labelFile), index=False)
        bayes_net = build_BN(labelFile, label.columns, savePng)
        if ensemble == "T":
            df = BCC_test_2_fold(data, label, dataPath, bayes_net, s, 10)
        else:
            df = BCC_test_2_fold(data, label, dataPath, bayes_net, s, 1)

        df_result = pd.concat([df_result, df], axis=1)

    df_result.columns = range(df_result.shape[1])

    # save the results
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    df_result.to_csv(savePath+dataset+".csv")