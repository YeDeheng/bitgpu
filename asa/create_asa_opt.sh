#!/bin/zsh

lines=`cat ${1} | wc -l`
sed -i "s/parameter_dimension.*/parameter_dimension\t\t\t$lines/" asa_opt.part1 

export LD_LIBRARY_PATH=../lib
../bin/prune_driver ${1} ${2} ${3}
cat asa_opt.part1 > asa_opt
cat ${1}.asa >> asa_opt
cat asa_opt.part2 >> asa_opt
