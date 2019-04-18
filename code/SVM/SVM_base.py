import pandas as pd
import numpy as np
import os
import time
import subprocess
import re
import random
import arff

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import hamming_loss
from sklearn import metrics
from sklearn.metrics import zero_one_loss
from sklearn.metrics import jaccard_similarity_score
from skmultilearn.model_selection import iterative_train_test_split
from pomegranate import BayesianNetwork
from collections import defaultdict

def evaluation(y_pred, y_prob, y_true):
    coverage = coverage_error(y_true, y_prob)
    hamming = hamming_loss(y_true, y_pred)
    ranking_loss = label_ranking_loss(y_true, y_prob)

    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro')

    acc = 0
    for i in range(y_true.shape[0]):
        acc += jaccard_similarity_score(y_true.iloc[i, :], y_pred.iloc[i, :])  # jaccard_similarity_score
    acc = acc / y_true.shape[0]

    zero_one = zero_one_loss(y_true, y_pred)  # 0-1 error

    performance = {"coverage_error": coverage,
                   "ranking_loss": ranking_loss,
                   "hamming_loss": hamming,
                   "f1_macro": f1_macro,
                   "f1_micro": f1_micro,
                   "Jaccard_Index": acc,
                   "zero_one_error": zero_one}
    return performance

# Class to represent a graph
class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)  # dictionary containing adjacency List
        self.V = vertices  # No. of vertices

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].extend(v)

        # A recursive function used by topologicalSort

    def topologicalSortUtil(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

                # Push current vertex to stack which stores result
        stack.insert(0, v)

        # The function to do Topological Sort. It uses recursive

    # topologicalSortUtil()
    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = [False] * self.V
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

                # Print contents of the stack
        return stack

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


def BR_test(X_train, X_test, y_train, y_test):
    # data set information
    n_label = y_test.shape[1]
    # split training and test data set

    X_train = pd.DataFrame(X_train, columns=X_train.columns)
    X_test = pd.DataFrame(X_test, columns=X_train.columns)

    y_train = pd.DataFrame(y_train, columns=y_train.columns)
    y_test = pd.DataFrame(y_test, columns=y_train.columns)

    # training
    classifier_list, training_time = naiveBayes_multi_label_training_BR(X_train, y_train)

    # testing
    y_predict, y_prob, testing_time = naiveBayes_multi_label_testing_BR(X_test, n_label, classifier_list)

    y_predict.columns = y_train.columns
    return y_predict, y_test


def predict_SVM(X_train, X_test, y_train_i):
    if y_train_i.nunique() != 1:
        clf = svm.SVC(kernel="linear", gamma='auto', probability=True)
        clf.fit(X_train, y_train_i)
        pred = clf.predict(X_test)
        prob = clf.predict_proba(X_test)[:, 1]

    else:
        pred = np.array([0] * X_test.shape[0])
        prob = np.array([0] * X_test.shape[0])
    return pred, prob


def build_bayes_net(trainLabelFile, labelName, savePath):
    cmd = """cd ../programme/Chordalysis/ 
    java -Xmx1g -classpath bin:lib/core/commons-math3-3.2.jar:lib/core/jayes.jar:lib/core/jgrapht-jdk1.6.jar:lib/extra/jgraphx.jar:lib/loader/weka.jar demo.Run %s 0.05 %s false
    """ % (trainLabelFile, savePath + "bayes_net.png")

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    out, err = p.communicate()
    for line in out.splitlines():
        if line.decode("utf-8").startswith('['):
            graph_set = [i for i in map(lambda x: x.split(','),
                                        line.decode("utf-8").replace(' ', ',').strip('[[\,]]').split(',]['))]

    dic = {}
    for l in labelName:
        s = set()
        for i in map(lambda x: set(x) if l in x else None, graph_set):
            if i != None:
                s.update(i)
        s.remove(l)
        dic[l] = s

    return dic


## function for tree structure
def get_structure(model, labels):
    dic = {}
    for item, attr in zip(model.structure, labels):
        if item == ():
            dic[attr] = {}
        else:
            dic[attr] = set(labels[list(item)])
    return dic


def get_order(model, labels):
    g = Graph(len(labels))
    for item, i in zip(model.structure, range(len(labels))):
        if item == ():
            pass
        else:
            g.addEdge(i, list(item))

    # get order
    a = g.topologicalSort()
    a.reverse()

    return labels[a]


def get_order_bayesnet(bayes_net, root):
    visited = []
    for key, value in bayes_net.items():
        if value == {}:
            visited.append(key)


    open_l = [root]
    while open_l != []:
        root = open_l.pop(0)
        if root not in visited:
            visited.append(root)
            open_l.extend(list(bayes_net[root]))

    return visited


def ClassifierChain_SVM(data_train, data_test, label_train, label_test, savePath, ensemble=1):
    n_label = label_train.shape[1]
    # for storing ensemble results
    pred_ensemble = pd.DataFrame(np.zeros(label_test.shape), columns=label_test.columns)
    prob_ensemble = pd.DataFrame(np.zeros(label_test.shape), columns=label_test.columns)

    # for loop for ensembling.
    for i in range(ensemble):
        X_train, X_test, y_train, y_test = data_train, data_test, label_train, label_test

        # create a random order.
        order = random.sample(list(range(n_label)), n_label)  # get orders
        for index in order:
            label = y_train.columns[index]  # the label to be fitted.
            y_train_i = y_train.loc[:, label]
            y_test_i = y_test.loc[:, label]

            pred, prob = predict_SVM(X_train, X_test, y_train_i)
            pred = pd.Series(pred, name=y_train_i.name)
            prob = pd.Series(prob, name=y_train_i.name)
            pred_ensemble.loc[:, label] += pred
            prob_ensemble.loc[:, label] += prob

            # add the prediction to the attribute matrix.
            X_train = pd.concat([X_train, y_train_i], axis=1)
            X_test = pd.concat([X_test, pred], axis=1)

    pred_ensemble = (((pred_ensemble / ensemble) >= 0.5) * 1).astype('int')
    prob_ensemble = prob_ensemble / ensemble
    pred_ensemble = pred_ensemble.fillna(0)
    prob_ensemble = prob_ensemble.fillna(0)

    return pred_ensemble, prob_ensemble


def BayesianClassifierChain_SVM(data_train, data_test, label_train, label_test, savePath,
                               ensemble=1, ordering="random", structure="bayes_net", lead=False):
    n_label = label_train.shape[1]
    # for storing ensemble results
    pred_ensemble = pd.DataFrame(np.zeros(label_test.shape), columns=label_test.columns)
    prob_ensemble = pd.DataFrame(np.zeros(label_test.shape), columns=label_test.columns)

    if ensemble == 1:
        # bayes_net structure
        if structure == "bayes_net":
            if lead:
                y_predict, y_true = BR_test(data_train, data_test, label_train, label_test)
                error_matrix = pd.DataFrame(np.array(y_predict) - np.array(y_true), columns=label_test.columns)
                error_matrix.to_csv(savePath + "y_train.csv", index=False)  # for learning bayes_net structure
            else:
                label_train.to_csv(savePath + "y_train.csv", index=False)  # for learning bayes_net structure
            bayes_net = build_bayes_net(os.path.abspath(savePath + "y_train.csv"), label_train.columns,
                                        os.path.abspath(savePath)+"/")

            # ordering
            if ordering == "best_prediction":
                y_pred, y_test = BR_test(data_train, data_test, label_train, label_test)
                acc = (y_pred.values == y_test.values).mean(axis=0)
                order = list(label_train.columns[np.argsort(-acc)])

            elif ordering == "most_edges":
                a = [(x, len(y)) for x, y in bayes_net.items()]
                a_sort = sorted(a, key=lambda x: x[1], reverse=True)
                root = [x[0] for x in a_sort]
                order = get_order_bayesnet(bayes_net, root)

            elif ordering == "random":
                root = label_train.columns[random.randint(0, len(label_train.columns) - 1)]
                order = get_order_bayesnet(bayes_net, root)

            else:
                raise (ValueError, "ordering should be one of {random, best_prediction, most_edges}")

        # tree structure
        elif structure == "tree":
            if ordering == "random":
                root = random.randint(0, len(label_train.columns) - 1)
                model = BayesianNetwork.from_samples(label_train, algorithm='chow-liu', root=root)
                bayes_net = get_structure(model, label_train.columns)
                order = get_order(model, label_train.columns)
            else:
                raise ValueError("in tree structure, only random ordering is applied.")

        else:
            raise ValueError("structure should be one of {bayes_net, tree}")

        # BayesianClassifierChain without ensemble.
        X_train, X_test, y_train, y_test = data_train, data_test, label_train, label_test
        learned_label = []
        for label in order:
            par = [x for x in bayes_net[label] if x in learned_label]

            X_tr = pd.concat([X_train, y_train.loc[:, par]], axis=1)
            X_te = pd.concat([X_test, pred_ensemble.loc[:, par]], axis=1)

            y_train_i = y_train.loc[:, label]
            y_test_i = y_test.loc[:, label]

            pred, prob = predict_SVM(X_tr, X_te, y_train_i)

            pred = pd.Series(pred, name=y_train_i.name)
            prob = pd.Series(prob, name=y_train_i.name)
            pred_ensemble.loc[:, label] += pred
            prob_ensemble.loc[:, label] += prob

            learned_label.append(label)

        pred_ensemble = pred_ensemble.fillna(0)
        prob_ensemble = prob_ensemble.fillna(0)

        return pred_ensemble, prob_ensemble

    else:
        # for loop for ensembling.
        for i in range(ensemble):
            # get bayesian network structure with Chordalysis.
            if structure == "bayes_net":
                if lead:
                    y_predict, y_true = BR_test(data_train, data_test, label_train, label_test)

                    error_matrix = pd.DataFrame(np.array(y_predict) - np.array(y_true), columns=label_test.columns)
                    error_matrix.to_csv(savePath + "y_train.csv", index=False)  # for learning bayes_net structure
                else:
                    label_train.to_csv(savePath + "y_train.csv", index=False)  # for learning bayes_net structure
                bayes_net = build_bayes_net(os.path.abspath(savePath + "y_train.csv"), label_train.columns,
                                            os.path.abspath(savePath) + "/")

                if ordering == "random":
                    root = label_train.columns[random.randint(0, len(label_train.columns) - 1)]
                    order = get_order_bayesnet(bayes_net, root)
                else:
                    raise ValueError("random!")

            elif structure == "tree":
                if ordering == "random":
                    root = random.randint(0, len(label_train.columns) - 1)
                    model = BayesianNetwork.from_samples(label_train, algorithm='chow-liu', root=root)
                    bayes_net = get_structure(model, label_train.columns)
                    order = get_order(model, label_train.columns)
                else:
                    raise ValueError("in tree structure, only random ordering is applied.")

            else:
                raise ValueError("structure should be one of {bayes_net, tree}")

            # BayesianClassifierChain with ensemble.
            X_train, X_test, y_train, y_test = data_train, data_test, label_train, label_test
            learned_label = []
            for label in order:
                par = [x for x in bayes_net[label] if x in learned_label]

                X_tr = pd.concat([X_train, y_train.loc[:, par]], axis=1)
                X_te = pd.concat([X_test, pred_ensemble.loc[:, par]], axis=1)

                y_train_i = y_train.loc[:, label]
                y_test_i = y_test.loc[:, label]

                pred, prob = predict_SVM(X_tr, X_te, y_train_i)

                pred = pd.Series(pred, name=y_train_i.name)
                prob = pd.Series(prob, name=y_train_i.name)
                pred_ensemble.loc[:, label] += pred
                prob_ensemble.loc[:, label] += prob

                learned_label.append(label)

        pred_ensemble = (((pred_ensemble / ensemble) >= 0.5) * 1).astype('int')
        prob_ensemble = prob_ensemble / ensemble
        pred_ensemble = pred_ensemble.fillna(0)
        prob_ensemble = prob_ensemble.fillna(0)
        return pred_ensemble, prob_ensemble


def two_fold(methods, data, label, dataset, ensemble=1, ordering="random", structure="bayes_net", lead=False):
    # setup
    savePath = "../code/temp/" + methods.__name__ + "/" + dataset + "/"
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    print("running", methods.__name__)
    print("setting:", ensemble, ordering, structure, lead)
    performance_df_all = pd.DataFrame()
    for j in range(5):
        print("time:", j)
        X_train, y_train, X_test, y_test = iterative_train_test_split(np.matrix(data), np.matrix(label), test_size=0.5)
        X_train = pd.DataFrame(X_train, columns=data.columns)
        X_test = pd.DataFrame(X_test, columns=data.columns)
        y_train = pd.DataFrame(y_train, columns=label.columns)
        y_test = pd.DataFrame(y_test, columns=label.columns)

        for i in range(2):
            X_test, X_train = X_train, X_test
            y_test, y_train = y_train, y_test

            # test
            if methods.__name__ == "BayesianClassifierChain_SVM":
                pred_ensemble, prob_ensemble = BayesianClassifierChain_SVM(X_train, X_test, y_train, y_test, savePath,
                                                                             ensemble, ordering, structure, lead)
            elif methods.__name__ == "ClassifierChain_SVM":
                pred_ensemble, prob_ensemble = ClassifierChain_SVM(X_train, X_test, y_train, y_test,
                                                                     savePath, ensemble)

            else:
                raise BaseException("no such a function")

            performance = evaluation(pred_ensemble, prob_ensemble, y_test)
            performance_df = pd.DataFrame.from_dict(performance, orient='index')
            performance_df_all = pd.concat([performance_df_all, performance_df], axis=1)
    performance_df_all.columns = list(range(10))
    return performance_df_all

if __name__ == "__main__":
    pass