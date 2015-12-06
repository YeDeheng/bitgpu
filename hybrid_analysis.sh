#!/bin/zsh -i

rm -f ./data/hybrid.dat
rm -f ./data/hybrid_speedup.dat
echo "design, cpu_time, gpu_time, cpu10, gpu10, cpu25, gpu25, cpu50, gpu50, cpu100, gpu100, cpu300, gpu300 " > ./data/hybrid.dat
echo "design, gpu, gpu10, gpu25, gpu50, gpu100, gpu300" > ./data/hybrid_speedup.dat

for DESIGN in poly poly6 diode level1_linear level1_satur approx1 approx2 iir8 caprasse BlackScholes binomial_option fir32 bellido
do
	cput=`cat ./data/quality_time_asa_$DESIGN.dat | cut -f 2 -d','`
	gput=`cat ./data/quality_time_bitslice_$DESIGN.dat | cut -f 2 -d','`
	speedupt=`echo "scale=2; $cput/$gput" | bc -l`
	echo -n $DESIGN, $cput, $gput >> ./data/hybrid.dat
	echo -n $DESIGN, $speedupt >> ./data/hybrid_speedup.dat

	for LIMIT in 10 25 50 100 300 
	do
		hybridt0=`cat ./data/quality_time_hybrid_asa_$DESIGN.dat | grep "^$LIMIT," | cut -f 3 -d','`
		hybridt1=`cat ./data/quality_time_hybrid_$DESIGN.dat | grep "^$LIMIT," | cut -f 3 -d','`
		speedup=`echo "scale=2; $cput/($hybridt0+$hybridt1)" | bc -l`
		echo -n ,$hybridt0, $hybridt1 >> ./data/hybrid.dat
		echo -n ,$speedup >> ./data/hybrid_speedup.dat
	done
	echo "" >> ./data/hybrid.dat
	echo "" >> ./data/hybrid_speedup.dat
done

