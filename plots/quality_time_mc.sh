#!/bin/zsh
echo "#design   MC_size   #MC time (ms)   quality" >  quality_time_mc.dat

#MC=65536
for design in poly4 # poly6 poly8 #diode level1_linear level1_satur approx1 approx2 rgb dct 
do
    for MC in 15 16 17 18 19 20 21 22 23 24 #8192 16384 32768 65536 131072
    do
            sed -i "451s/pow(2,.*)/pow(2,$MC)/" src/peace_test.cu
            echo -n "$design   $MC   " >> quality_time_mc.dat

            runtime=`make DESIGN=$design INTERVALS=1 METHOD=mc | grep "kernel time is" | cut -d" " -f5`
            echo -n "$runtime   " >> quality_time_mc.dat
            make DESIGN=$design INTERVALS=1 METHOD=mc | grep "minimum area is" | cut -d" " -f4 >> quality_time_mc.dat
    done
done
