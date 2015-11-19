#!/bin/zsh
output=quality_time_mc.dat
echo "#design,MC_size,time(ms),area_min,area_max,area_mean" > $output
source ~/.zshrc
touch mymean_tmp
#MC=65536
for design in fig3 # poly6 poly8 #diode level1_linear level1_satur approx1 approx2 rgb dct 
do
    for MC in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 #8192 16384 32768 65536 131072
    do
        n=0
        area_min=100000
        area_max=0
        area_all=0
        rm mymean_tmp
        samples=`echo "2 ^ $MC" | bc`
        for i in 1 2 3 4 5 6 7 8 9 10
        do
            runtime=`make DESIGN=$design INTERVALS=1 METHOD=mc CONFIG=$samples | grep "kernel time is" | cut -d" " -f5`
            area=`make DESIGN=$design INTERVALS=1 METHOD=mc CONFIG=$samples | grep "minimum area is" | cut -d" " -f4`
            if [ $area -ne 100000 ]
            then
                echo "scale=2; $area/100" | bc >> mymean_tmp
                n=`echo $n + 1 | bc`
                area_all=`expr $area_all + $area`
                if [ $area -gt $area_max ]
                then
                    area_max=$area
                fi
                if [ $area -lt $area_min ]
                then
                    area_min=$area
                fi
            fi
        done
            mean=`cat mymean_tmp | mymean`
            mean=`echo "$mean * 100" | bc`
            area_aver=`echo "scale=2; $area_all/$n" | bc`
        echo "$design,$samples,$runtime,$area_min,$area_max,$area_aver,$mean" >> $output
    done
done
