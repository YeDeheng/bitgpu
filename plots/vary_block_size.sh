#!/bin/zsh
echo "#design   MC_size   block_size   #MC time (ms)" > vary_MC_blocksize_all.dat

#MC=65536
for design in fig3 poly4 poly6 poly8 #diode level1_linear level1_satur approx1 approx2 rgb dct 
do
    for MC in 8192 16384 32768 65536 131072
    do
        for block_size in 32 96 128 160 224 480 512 576 640 736 800 928 1024
            #32 64 96 128 160 192 224 256 278 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 1024 
        do
            #MC=`echo $blocks \\* $block_size | bc`
            sed -i "514s/block_size = .*;/block_size = $block_size;/" src/peace_test.cu
            sed -i "515s/sampling_cnt = .*;/sampling_cnt = $MC;/" src/peace_test.cu
            echo -n "$design   $MC   $block_size   " >> vary_MC_blocksize_all.dat

            make DESIGN=$design INTERVALS=1 | grep "Monte-carlo time is" | cut -d" " -f4 >> vary_MC_blocksize_all.dat
        done
    done
done
