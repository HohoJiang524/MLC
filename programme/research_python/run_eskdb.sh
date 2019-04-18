#!/bin/bash
dirPath=/Volumes/Samsung_T5/research/programme/ESKDB_HDP/result/$1
filePath=$6
mkdir $dirPath
cd /Volumes/Samsung_T5/research/programme/ESKDB_HDP
for file in $filePath/*
do
    filename=$(basename $file .arff).txt
    echo running $filename ...
    java -jar target/ESKDBHDP-0.0.1-SNAPSHOT-jar-with-dependencies.jar -t $file -S ESKDB -K $2 -I $3 -L $4 -E $5 -V -M \
    > $dirPath$'/'$filename
done
