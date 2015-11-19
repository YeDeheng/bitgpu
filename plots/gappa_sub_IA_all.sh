#!/bin/zsh
output=~/interval_gpu/plots/gappa_sub_IA_all.csv
echo "#benchmark,#sub_range,gappa_runtime" > $output 
sub_ranges=1000

pushd ../
for design in fig3 appolonoius caprasse poly4 poly6 poly8 diode level1_linear level1_satur approx1 approx2 #rgb dct
do
    (time make gappa_sub_IA DESIGN=$design INTERVALS=$sub_ranges) &> gappa_sub_IA.log
    runtime=`cat gappa_sub_IA.log | grep "total" | cut -d" " -f12`
    echo "$design,$sub_ranges,$runtime" >> $output
done
rm gappa_sub_IA.log
popd
