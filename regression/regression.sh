#!/bin/zsh


#for design in mult #sub #mult div exponential logarithm
for design in square_root
do
    cd $design
    echo "bits, slices, ffs, luts, brams, dsps, cp achieved, cp requested" > bit_sweep_${design}.csv
    for bits in 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80
    do
	sed -i "5s/ap_fixed<.*,10>/ap_fixed<$bits,10>/g" $design.cpp
	echo -n "$bits    " >> bit_sweep_${design}.csv
	make -f Makefile.$design autoesl
	make -f Makefile.$design hardware
	for par in SLICE FF LUT BRAM DSP "CP required" "CP achieved"
	do
	    rpt=`cat ${design}_batch.prj/solution1/impl/report/vhdl/${design}_export.rpt | grep "$par" | cut -d":" -f 2 | sed "s/\s//g"`
	    echo -n $rpt >> bit_sweep_${design}.csv
	    echo -n "    " >> bit_sweep_${design}.csv
	done
	echo >> bit_sweep_${design}.csv
    done
    cd ..
done
