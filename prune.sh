#!/bin/zsh -i

UPPER_BOUND=25
ERROR_THRESH=2e-4

for DESIGN in poly diode level1_linear bellido poly6 level1_saturation approx1 approx2 poly8 caprasse 
do
	echo -n "benchmark:  "
	echo $DESIGN
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib
	echo -n "search space after pruning:  "
	./bin/prune_driver ./data/$DESIGN/$DESIGN.asm $ERROR_THRESH $UPPER_BOUND  | grep search | sed "s/.*:\(.*\)/$DESIGN \1/"
done

