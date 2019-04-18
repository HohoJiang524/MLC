#!/bin/bash

cd code

# dataset is the name of dataset
# T or F to control whether an ensemble way to use.

dataset=$1

# NB
echo Naive Bayes
echo ClassifierChain_NB
python3 ./NB/ClassifierChain_NB_bayesNet.py $dataset 

# echo EnsembleClassifierChain_NB
#python3 ./NB/EnsembleClassifierChain_NB.py $dataset

echo BayesianClassifierChain_NB
python3 ./NB/BayesianClassifierChain_NB_bayesNet.py $dataset

echo LEAD_NB
python3 ./NB/LEAD_NB_bayesNet.py $dataset

# SVM
echo SVM
echo ClassifierChain_SVM
python3 ./SVM/ClassifierChain_SVM_bayesNet.py $dataset 

# echo EnsembleClassifierChain_SVM
#python3 ./SVM/EnsembleClassifierChain_SVM.py $dataset

echo BayesianClassifierChain_SVM
python3 ./SVM/BayesianClassifierChain_SVM_bayesNet.py $dataset

echo LEAD_SVM
python3 ./SVM/LEAD_SVM_bayesNet.py $dataset

# different ordering
echo ordering
echo best_prediction_NB_BCC
python3 ./orders/best_prediciton_NB_BCC_bayesNet.py $dataset

echo most_edges_NB_BCC
python3 ./orders/most_edges_NB_BCC_bayesNet.py $dataset

echo random_NB_BCC
python3 ./orders/random_NB_BCC_bayesNet.py $dataset

# different structure
echo structure
echo bayesNet_NB
python3 ./structure/bayesNet_NB_BCC.py $dataset

echo tree_NB
python3 ./structure/tree_NB_BCC.py $dataset

# ESKDB
echo ESKDB

echo ClassifierChain_ESKDB
python3 ./ESKDB/ClassifierChain_ESKDB_bayesNet.py $dataset real

echo BayesianClassifierChain_ESKDB_bayesNet
python3 ./ESKDB/BayesianClassifierChain_ESKDB_bayesNet.py $dataset real

echo LEAD_ESKDB
python3 ./ESKDB/LEAD_ESKDB_bayesNet.py $dataset real

echo FINISH.


