# -*- coding: utf-8 -*-

from eskdb_base import two_fold
from eskdb_base import ClassifierChain_ESKDB
import pandas as pd
import os
import sys

if __name__ == "__main__":
    dataset = sys.argv[1]
    datatype = sys.argv[2]
    # setup
    savePath = "../result/ESKDB/CC_ESKDB_bayesNet/"
    dataPath = os.path.abspath("../data/" + dataset + "/")
    X_file = "X_scale.csv"
    y_file = "y.csv"

    data = pd.read_csv(os.path.join(dataPath, X_file))
    label = pd.read_csv(os.path.join(dataPath, y_file))

    df_CC = two_fold(ClassifierChain_ESKDB, data, label, dataset, datatype,
                     ensemble=1, ordering="random", structure="bayes_net", lead=False)

    # save the results
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    df_CC.to_csv(savePath+dataset+".csv")
