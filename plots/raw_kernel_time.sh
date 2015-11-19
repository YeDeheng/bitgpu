#!/bin/zsh
output=raw_kernel_time.csv
echo "model,operation,threads,block_size,runtime" > $output | cat 

# 0=range, 1=error, 2=area
for model in 0 1 2
do
	# 0=add, 1=sub, 2=mult, 3=div, 4=exp, 5=log
	for operation in 0 1 2 3 4 5
	do
		for threads in 131072 #65536 #512 4096 16384 65536 # 131072 524288 
		do
			for block_size in 32 64 128 256 512 1024 #128 320 512 800 1024 
			do
				../bin/raw_kernels_driver $model $operation $threads $block_size > /tmp/data.log
				time=`cat /tmp/data.log | grep "kernel time" | cut -d" " -f5`
				echo "$model,$operation,$threads,$block_size,$time" >> $output | cat 
			done
		done
	done
done
		
exit
for threads in 131072 
do
	for block_size in 32 42 128 #320 512 800 1024 
	do
		../bin/raw_kernels_driver 4 -1 $threads $block_size > /tmp/data.log
		time=`cat /tmp/data.log | grep "kernel time" | cut -d" " -f5`
		echo "4,-1,$threads,$block_size,$time" >> $output | cat 
	done
done
