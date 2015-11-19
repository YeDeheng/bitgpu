#!/bin/zsh
output=quality_time_mc.dat
echo "#design   MC_size   time(ms)   quality" > $output

#MC=65536
for design in fig3 # poly6 poly8 #diode level1_linear level1_satur approx1 approx2 rgb dct 
do
    for MC in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 #8192 16384 32768 65536 131072
    do
            sed -i "453s/pow(2,.*)/pow(2,$MC)/" src/peace_test.cu
            echo -n "$design   $MC   " >> $output

            runtime=`make DESIGN=$design INTERVALS=1 METHOD=mc | grep "kernel time is" | cut -d" " -f5`
            echo -n "$runtime   " >> quality_time_mc.dat
            make DESIGN=$design INTERVALS=1 METHOD=mc | grep "minimum area is" | cut -d" " -f4 >> $output
    done
done
