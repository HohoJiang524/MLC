# -*- coding: utf-8 -*-

from NB_base import two_fold
from NB_base import ClassifierChain_NB
import pandas as pd
import os
import sys

if __name__ == "__main__":
    dataset = sys.argv[1]

    # setup
    savePath = "../result/NB/ECC_NB_bayesNet/"
    dataPath = os.path.abspath("../data/" + dataset + "/")
    X_file = "X_scale.csv"
    y_file = "y.csv"

    data = pd.read_csv(os.path.join(dataPath, X_file))
    label = pd.read_csv(os.path.join(dataPath, y_file))

    df_CC = two_fold(ClassifierChain_NB, data, label, dataset,
                     ensemble=10, ordering="random", structure="bayes_net", lead=False)

    # save the results
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    df_CC.to_csv(savePath+dataset+".csv")
