#!/bin/zsh -i

UPPER_BOUND=25
ERROR_THRESH=2e-4

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib
rm -f ./data/quality_time_bitslice.dat
for DESIGN in level1_linear poly diode bellido poly6 level1_saturation approx1 poly8 approx2 caprasse
do
	echo $DESIGN
	./bin/range_driver ./data/$DESIGN/$DESIGN.asm 1 32
	./bin/prune_driver ./data/$DESIGN/$DESIGN.asm $ERROR_THRESH $UPPER_BOUND
	rm -f ./data/raw_bitslice_$DESIGN.dat
	for BLOCK in 32 64 128 256 512
	do
		./bin/bitslice_driver ./data/$DESIGN/$DESIGN.asm \
			$ERROR_THRESH \
			1 \
			1 \
			$BLOCK | grep -v "good" >> ./data/raw_bitslice_$DESIGN.dat
	done
	cat ./data/raw_bitslice_$DESIGN.dat | mymin > ./data/quality_time_bitslice_$DESIGN.dat
	cat ./data/raw_bitslice_$DESIGN.dat | mymin | sed "s/^/$DESIGN,/" > ./data/quality_time_bitslice.dat
done
