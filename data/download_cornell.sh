#! /bin/bash

for i in $(seq -f "%02g" 1 10)
do
    wget http://pr.cs.cornell.edu/grasping/rect_data/temp/data$i.tar.gz -P cornell/
    tar -xvzf cornell/data$i.tar.gz -C cornell/
    rm cornell/data$i.tar.gz
    mv cornell/$i/* cornell/
    rmdir cornell/$i
done
