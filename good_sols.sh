#!/bin/zsh -i

UPPER_BOUND=35

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib
rm -f ./data/good_sols.dat
for DESIGN in poly #poly6 poly8 diode level1_linear level1_satur approx1 approx2
do
	./bin/range_driver ./data/$DESIGN/$DESIGN.asm 1
	for ERROR_THRESH in 1 2e-1 2e-2 2e-3 2e-4 2e-5 2e-6
	do
	./bin/prune_driver ./data/$DESIGN/$DESIGN.asm $ERROR_THRESH $UPPER_BOUND
	./bin/bitslice_driver ./data/$DESIGN/$DESIGN.asm \
			$ERROR_THRESH \
			1 \
			1 \
			128 | grep "good" | sed "s/^good,/$DESIGN,$ERROR_THRESH,/" >> ./data/good_sols.dat |cat
	done
done
