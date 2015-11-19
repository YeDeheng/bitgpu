#!/bin/zsh
echo "#design   Nb-Na   bf_time   MC_time (ms)" > compare_mc_bf.dat

#MC=65536
for design in fig3 
do
    for range in 8 9 10 11 12 13 14 15 16
    do
            Nb=`echo 6 + $range | bc`
            sed -i "376s/= .*;/= $Nb;/" src/peace_test.cu
            sed -i "451s/pow(.*)/pow($range,6)/" src/peace_test.cu
            echo -n "$design   $range   " >> compare_mc_bf.dat

            make clean
            bf_time=`make DESIGN=$design INTERVALS=1 METHOD=bit_slice | grep "kernel time is" | cut -d" " -f5`
            echo -n "$bf_time   " >> compare_mc_bf.dat
            make clean 
            make DESIGN=$design INTERVALS=1 METHOD=mc | grep "kernel time is" | cut -d" " -f5 >> compare_mc_bf.dat
    done
done
