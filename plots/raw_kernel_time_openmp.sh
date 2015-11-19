#!/bin/zsh
output=raw_kernel_time_openmp.csv
echo "model,operation,loop_count,threads,runtime" > $output | cat 

# 0=range, 1=error, 2=area
for model in 0 1 2
do
	# 0=add, 1=sub, 2=mult, 3=div, 4=exp, 5=log
	for operation in 0 1 2 3 4 5
	do
		for loop_count in 131072 #65536 #512 4096 16384 65536 # 131072 524288 
		do
			for threads in 1 2 4 8
			do
				../bin/raw_kernels_openmp_driver $model $operation $loop_count $threads > /tmp/data.log
				time=`cat /tmp/data.log | grep "kernel time" | cut -d" " -f5`
				echo "$model,$operation,$loop_count,$threads,$time" >> $output | cat 
			done
		done
	done
done
		
exit
for loop_count in 131072 #65536 #512 4096 16384 65536 # 131072 524288 
do
	for threads in 1 2 4 8
	do
		../bin/raw_kernels_openmp_driver 4 -1 $loop_count $threads > /tmp/data.log
		time=`cat /tmp/data.log | grep "kernel time" | cut -d" " -f5`
		echo "4,-1,$loop_count,$threads,$time" >> $output | cat 
	done
done
