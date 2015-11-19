#!/bin/zsh -i

rm -f ./data/sizes.dat
for DESIGN in poly poly6 poly8 diode level1_linear level1_satur approx1 approx2 sobel gaussian
do
	lines=`wc -l ./data/$DESIGN/$DESIGN.asm | cut -d' ' -f 1`
	echo "$DESIGN,$lines">> ./data/sizes.dat
done
