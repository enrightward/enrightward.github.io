#!/bin/bash

inputfile=$1
imgdir="../assets/img/order-statistics/part1"
count=0

for file in "$imgdir"/*.png
do
    ((count+=1))    
    cat $inputfile | awk -v x="$file" -v y="$count" '/\[png/{c++; if (c==y) { sub("\\[png.*","[png]("x")")}}1'
done
