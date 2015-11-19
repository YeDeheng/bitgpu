#!/bin/zsh -i

# iir8 fir32
#for DESIGN in BlackScholes #poly poly6 poly8 diode level1_linear level1_satur approx1 approx2 caprasse BlackScholes binomial_option bellido
for DESIGN in poly iir8 deriche gaussian sobel
do
	pushd ./scripts
	./gimple_to_asm.sh ../bench/$DESIGN.c
	popd
done
