#!/bin/zsh -i

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib

echo "design,time,block_size,intervals"  > ../data/range.dat | cat
#for DESIGN in poly sobel # iir8 deriche gaussian sobel
for DESIGN in poly poly6 diode level1_linear level1_satur approx1 approx2 poly8 caprasse bellido sobel gaussian
do
	for SPLITS in 1 2 3 4 5 6 7 8 16 32 64 128 256 512 1024 2048 4096 8192
	do
		for BLOCKS in 32 64 128 256 512
		do
			echo $DESIGN $SPLITS $BLOCKS
			./bin/range_driver ./data/$DESIGN/$DESIGN.asm $SPLITS $BLOCKS > ./data/range_$DESIGN.dat
			cat ./data/range_$DESIGN.dat | sed "s/range,/$DESIGN,/" >> ./data/range.dat | cat
		done
	done
done
