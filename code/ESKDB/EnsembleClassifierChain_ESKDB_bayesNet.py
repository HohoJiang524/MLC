import eskdb_base
from eskdb_base import two_fold
from eskdb_base import ClassifierChain_ESKDB
import pandas as pd
import os
import sys
import shutil
if __name__ == "__main__":

    dataset = sys.argv[1]
    datatype = sys.argv[2]
    # setup
    savePath = "../result/ESKDB/ECC_ESKDB_bayesNet/"

    temp_path = "../code/temp/LEAD_ESKDB_bayesNet/" + dataset + "/"
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

    dataPath = '../data/' + dataset + "/"
    X_file = "X_scale.csv"
    y_file = "y.csv"

    # read data
    data = pd.read_csv(os.path.join(dataPath, X_file))
    label = pd.read_csv(os.path.join(dataPath, y_file))

    df_ECC = two_fold(ClassifierChain_ESKDB, data, label, dataset, datatype,
                     ensemble=10, ordering="random", structure="bayes_net", lead=False)

    # save the results
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    df_ECC.to_csv(savePath+dataset+".csv")