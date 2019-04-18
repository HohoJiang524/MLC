#!/bin/bash

for dataset in medical tmc2007 genbase enron;
do
	echo running $dataset
	./run_command_nominal.sh $dataset > $dataset.log 2>&1 & 
done