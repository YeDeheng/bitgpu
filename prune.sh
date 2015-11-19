#!/bin/zsh -i

UPPER_BOUND=25
ERROR_THRESH=2e-4

#for DESIGN in BlackScholes # poly poly6 poly8 diode level1_linear level1_satur approx1 approx2 caprasse bellido
#for DESIGN in poly diode level1_linear bellido poly6 level1_saturation approx1 approx2 poly8 caprasse 
for DESIGN in iir8 deriche gaussian sobel
do
	echo $DESIGN
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib
	./bin/prune_driver ./data/$DESIGN/$DESIGN.asm $ERROR_THRESH $UPPER_BOUND  | grep search | sed "s/.*:\(.*\)/$DESIGN \1/"
done

