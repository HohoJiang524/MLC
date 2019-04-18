#!/bin/bash
dirPath=/Volumes/Samsung_T5/research/programme/ESKDB_HDP/result/$1
fileArff=$6

if [ ! -d $dirPath ]; then
  mkdir $dirPath
fi

cd /Volumes/Samsung_T5/research/programme/ESKDB_HDP

filename=$(basename $fileArff .arff).txt
echo running $filename ...
java -jar target/ESKDBHDP-0.0.1-SNAPSHOT-jar-with-dependencies.jar -t $fileArff -S ESKDB -K $2 -I $3 -L $4 -E $5 -V -M \
> $dirPath$'/'$filename
