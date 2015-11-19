#!/bin/zsh
output=~/interval_gpu/plots/mc_all.csv
echo "#benchmark,samples,time(ms),area" > $output 

pushd ../
for design in fig3 poly4 poly6 poly8 diode level1_linear level1_satur approx1 approx2 #rgb dct
do
    for samples in 512 2048 8192 16384 32768 65536 131072 524288 1048576 2097152 4194304
    do
        echo -n "$design,$samples," >> $output
        runtime=`make DESIGN=$design INTERVALS=1 METHOD=mc CONFIG=${samples} | grep "kernel time is" | cut -d" " -f5`
        echo -n "$runtime," >> $output
        make DESIGN=$design INTERVALS=1 METHOD=mc CONFIG=${samples} | grep "minimum area is" | cut -d" " -f4 >> $output
    done
done
popd
