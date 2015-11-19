#!/bin/zsh
output=raw_bf_mc_asa.dat
echo "#design   Nb-Na   bf_time  bf_area   mc_time   mc_area" > $output

#MC=65536
for design in fig3 
do
    for Na in 9 8 7 6 5 4 3 2 1 0 
    do
            range=`echo 16 - $Na | bc`
            sed -i "377s/= .*;/= $Na;/" src/peace_test.cu
            sed -i "453s/pow(.*)/pow($range,6)/" src/peace_test.cu
            echo -n "$design   $range   " >> $output

            make clean
            bf_time=`make DESIGN=$design INTERVALS=1 METHOD=bit_slice | grep "kernel time is" | cut -d" " -f5`
            bf_area=`make DESIGN=$design INTERVALS=1 METHOD=bit_slice | grep "minimum area is" | cut -d" " -f4`
            echo -n "$bf_time   $bf_area   " >> $output
            mc_time=`make DESIGN=$design INTERVALS=1 METHOD=mc | grep "kernel time is" | cut -d" " -f5`
            mc_area=`make DESIGN=$design INTERVALS=1 METHOD=mc | grep "minimum area is" | cut -d" " -f4`
            echo  "$mc_time   $mc_area  " >> $output
    done
done
