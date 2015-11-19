#!/bin/zsh -i

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib
for DESIGN in poly poly6 diode level1_linear level1_satur approx1 approx2 poly8 caprasse bellido sobel gaussian
do
	echo $DESIGN
	./bin/stuff ./data/$DESIGN/$DESIGN.asm
done
