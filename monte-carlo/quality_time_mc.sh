#!/bin/zsh -i

UPPER_BOUND=25
ERROR_THRESH=2e-4

rm -f ./data/quality_time_mc.dat
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib
for DESIGN in poly poly6 poly8 diode level1_linear level1_satur approx1 approx2
do
	./bin/range_driver ./data/$DESIGN/$DESIGN.asm 1
	./bin/prune_mc_driver ./data/$DESIGN/$DESIGN.asm $ERROR_THRESH $UPPER_BOUND
	for SEED in 1 2 3 4 5
	do
		for SAMPLES in 1 10 1000 1000000 #10000000
		do
			./bin/monte_carlo_driver ./data/$DESIGN/$DESIGN.asm \
				$ERROR_THRESH \
				$UPPER_BOUND \
				$SAMPLES \
				1 \
				128 \
				$SEED | sed "s/^/$DESIGN,$SEED,$SAMPLES,/" >> ./data/quality_time_mc.dat
		done
	done
done
