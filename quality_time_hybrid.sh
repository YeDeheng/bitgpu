#!/bin/zsh -i

UPPER_BOUND=25
ERROR_THRESH=2e-4

for DESIGN in level1_linear poly diode bellido poly6 level1_saturation approx1 poly8 approx2 caprasse sobel gaussian
do
	echo $DESIGN

	rm -f ./data/quality_time_hybrid_$DESIGN.dat
	rm -f ./data/quality_time_hybrid_asa_$DESIGN.dat

	for LIMIT in 10 25 50 100 300
	do
		# get ASA derived ranges first
		pushd asa
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../lib
		./create_asa_opt.sh ../data/$DESIGN/$DESIGN.asm $ERROR_THRESH $UPPER_BOUND > /dev/null
		./asa_run ../data/$DESIGN/$DESIGN.asm $ERROR_THRESH $LIMIT | grep -v PART | tail -n 2 | head -n 1 | sed "s/^/$LIMIT,/" >> ../data/quality_time_hybrid_asa_$DESIGN.dat
		./asa_run ../data/$DESIGN/$DESIGN.asm $ERROR_THRESH $LIMIT | grep PART | sed "s/PART\ //g" > ../data/$DESIGN/$DESIGN.asm.bitrange.csv
		popd

		# then use that as starting point for GPU exploration
		rm -f ./data/raw_hybrid_$DESIGN.dat
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib
		for BLOCK in 32 64 128 256 512
		do
			./bin/bitgpu_driver ./data/$DESIGN/$DESIGN.asm \
				$ERROR_THRESH \
				1 \
				1 \
				$BLOCK | grep -v "good" >> ./data/raw_hybrid_$DESIGN.dat
		done
		cat ./data/raw_hybrid_$DESIGN.dat | mymin | sed "s/^/$LIMIT,/" >> ./data/quality_time_hybrid_$DESIGN.dat
	done
done

