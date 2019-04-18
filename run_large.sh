#!/bin/bash

for dataset in rcv1subset1 rcv1subset2 rcv1subset3 rcv1subset4 rcv1subset5;
do
	echo running $dataset
	./run_command_numeric.sh $dataset > $dataset.log 2>&1 & 
done