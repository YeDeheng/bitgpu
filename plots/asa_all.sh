#!/bin/zsh
output=~/interval_gpu/plots/asa_all.csv
echo "#benchmark,time(s),area" > $output 

pushd ../
for design in poly20 fir32 #fig3 poly4 poly6 poly8 diode level1_linear level1_satur approx1 approx2 #rgb dct
do
    make test DESIGN=$design
    (time make asa DESIGN=$design) &> runtime.log
    runtime=`cat runtime.log | grep total | cut -d" " -f11`
    area=`cat runtime.log | grep "final cost"| cut -d"=" -f2 | sed "s/\s//g"`
    echo "$design,$runtime,$area" >> $output
done
rm runtime.log
popd
