#!/bin/zsh -i

UPPER_BOUND=128

rm -f ./data/thresh.dat
#for DESIGN in poly bellido caprasse
for DESIGN in level1_linear poly diode bellido poly6 level1_saturation approx1 poly8 approx2 caprasse
do
	rm -f ../data/thresh_$DESIGN.dat
	for ERROR_THRESH in 2e-1 2e-2 2e-3 2e-4 2e-5 2e-6 2e-7
	do
		ERROR_THRESH2=`echo "$ERROR_THRESH * 1.8" | sed "s/e/*10^/" | bc -l`
		echo $DESIGN $ERROR_THRESH2
		pushd asa
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../lib

		./create_asa_opt.sh ../data/$DESIGN/$DESIGN.asm $ERROR_THRESH $UPPER_BOUND
		./asa_run ../data/$DESIGN/$DESIGN.asm \
			$ERROR_THRESH2 10 >> ../data/thresh_${DESIGN}_$ERROR_THRESH.dat
		cat ../data/thresh_${DESIGN}_${ERROR_THRESH}.dat | mymin | sed "s/^/$DESIGN,$ERROR_THRESH,/">> ../data/thresh.dat
		popd
	done
done
