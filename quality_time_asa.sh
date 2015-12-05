#!/bin/zsh -i

UPPER_BOUND=128
ERROR_THRESH=2e-4

rm -f ./data/quality_time_asa.dat
echo "bench,cost,time" > ./data/quality_time_asa.dat
for DESIGN in level1_linear poly diode bellido poly6 level1_saturation approx1 poly8 approx2 caprasse sobel gaussian 
do
	echo $DESIGN starting...
	pushd asa
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../lib
	./create_asa_opt.sh ../data/$DESIGN/$DESIGN.asm $ERROR_THRESH $UPPER_BOUND > /dev/null
	./asa_run ../data/$DESIGN/$DESIGN.asm $ERROR_THRESH 100000000 > ../data/raw_asa_$DESIGN.dat
	# need to record the termination time with tail -n 1
	cat ../data/raw_asa_$DESIGN.dat | tail -n 1 > ../data/quality_time_asa_$DESIGN.dat
	cat ../data/raw_asa_$DESIGN.dat | sed "s/^/$DESIGN,/" | cut -d"," -f1,2,3 >> ../data/quality_time_asa.dat
	popd
	echo $DESIGN finishing...
done

