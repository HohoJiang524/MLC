# -*- coding: utf-8 -*-

import pandas as pd
import arff
import os


def arff2csv(filePath):
    # read arff
    with open(filePath, 'r') as f:
        obj = arff.load(f)

    data = obj['data']
    # relation = obj['relation']
    # description = obj['description']

    # get column names
    column_name = list(map(lambda x: x[0], obj['attributes']))

    # write data to csv file
    filePath = filePath[:-4] + '.csv'

    df = pd.DataFrame(data, columns=column_name)

    df.to_csv(filePath, index=False)

    return df


def convert_all_arff(path):
    for file in os.listdir(path):
        if file[-4:] == 'arff':
            arff2csv(os.path.join(path, file))

        elif os.path.isdir(os.path.join(path, file)):
            print('\n.. open dir', os.path.join(path, file), '..')
            convert_all_arff(os.path.join(path, file))


def __main__():
    path = r'/Users/jiangjunhao/Desktop/ESKDB_HDP/data/small_datasets/'  # directory
    convert_all_arff(path)


