#!/bin/zsh
for design in poly6 poly8 diode level1_linear level1_satur approx1 approx2 # poly20 fir32
do
    output=${design}_quality_time_mc.dat
    echo "#time(ms)  mc_area" > $output
    for MC_loops in 1 64 # 128 256 512
    do
        for MC_size in 16 17 18 19 20 21 
        do
            samples_iter=`echo "2 ^ $MC_size" | bc`
            samples_total=`echo "$samples_iter * $MC_loops" | bc`
            runtime=`make DESIGN=$design INTERVALS=1 METHOD=mc CONFIG=$samples_iter MC_LOOPS=$MC_loops | grep "kernel time is" | cut -d" " -f5`
            area=`make DESIGN=$design INTERVALS=1 METHOD=mc CONFIG=$samples_iter MC_LOOPS=$MC_loops| grep "minimum area is" | cut -d" " -f4`
            echo "$runtime   $area" >> $output
        done
    done
    cp $output ./plots/$design/
done
