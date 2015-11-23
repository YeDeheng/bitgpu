#!/bin/zsh -i

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib

rm -f ./data/range_gappa.dat
pushd scripts > /dev/null
#for DESIGN in iir8 deriche gaussian sobel
echo "design,splits,runtime(us)"
for DESIGN in poly #poly6 diode level1_linear level1_satur approx1 approx2 poly8 caprasse bellido sobel gaussian
do
	for SPLITS in 1 8 16 32 64 128 256 512 1024 2048 4096 8192
	do
		./asm_to_gappa_sub_IA.sh ../bench/$DESIGN.c $SPLITS;
#		(time gappa ../data/$DESIGN/${DESIGN}_sub_IA.g) 2> range_gappa_$DESIGN.log 
		gappa ../data/$DESIGN/${DESIGN}_sub_IA.g 2> range_gappa_$DESIGN.log
		runtime=`cat range_gappa_$DESIGN.log | grep "Time" | sed "s/.*:\(.*\)us/\1/"`
		echo $DESIGN,$SPLITS,$runtime >> ../data/range_gappa_$DESIGN.dat
		echo $DESIGN,$SPLITS,$runtime >> ../data/range_gappa.dat | cat
	done
done
popd  > /dev/null
