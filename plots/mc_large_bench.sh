#!/bin/zsh
output=compare_asa_mc.dat1
echo "#design   samples_total  time(ms)  mc_area" > $output
MC_size=20
for design in poly20 fir32
do
    for MC_loops in 1 2 8 16 32 64 128 256 512
    do
        samples_iter=`echo "2 ^ $MC_size" | bc`
        samples_total=`echo "$samples * $MC_loops" | bc`
        runtime=`make DESIGN=$design INTERVALS=1 METHOD=mc CONFIG=$samples MC_LOOPS=$MC_loops | grep "kernel time is" | cut -d" " -f5`
        area=`make DESIGN=$design INTERVALS=1 METHOD=mc CONFIG=$samples MC_LOOPS=$MC_loops| grep "minimum area is" | cut -d" " -f4`
        echo "$design   $samples_total   $runtime   $area" >> $output
    done
done
