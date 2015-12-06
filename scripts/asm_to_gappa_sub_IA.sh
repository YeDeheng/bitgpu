#!/bin/zsh -i

# This script will translate the ASM format into a syntax suitable for Gappa evaluation
# The goal is to run sub-interval analysis to generate tighter bounds.
# This can be converted into a compiler optimiztion to make it more robust

source ~/.zshrc

touch dummy.svg dummy.gimple dummy.asm dummy.bak dummy.o dummy.tmp dummy.g
rm -f *.svg 2&> /dev/null
rm -f *.gimple* 2&> /dev/null
rm -f *.asm* 2&> /dev/null
rm -f *.bak* 2&> /dev/null
rm -f *.o 2&> /dev/null

file=`basename $1 .c`

pushd ../data/$file &> /dev/null
touch dummy.g dummy.tmp
rm -f *.g 2&> /dev/null  
rm -f *.tmp 2&> /dev/null

touch ${file}_sub_IA.g

# handle compute 
touch instr.tmp
cat $file.asm.bak | grep "ADD\|SUB\|MUL\|DIV\|EXP\|LOG" > instr.tmp
sed -i "s/ADD,\(.*\),\(.*\),\(.*\);/var_\1 = var_\2 + var_\3;/g" instr.tmp
sed -i "s/SUB,\(.*\),\(.*\),\(.*\);/var_\1 = var_\2 - var_\3;/g" instr.tmp
sed -i "s/MUL,\(.*\),\(.*\),\(.*\);/var_\1 = var_\2 * var_\3;/g" instr.tmp
sed -i "s/DIV,\(.*\),\(.*\),\(.*\);/var_\1 = var_\2 \/ var_\3;/g" instr.tmp
sed -i "s/EXP,\(.*\),\(.*\),\(.*\);/var_\1 = exp (var_\2);/g" instr.tmp
sed -i "s/LOG,\(.*\),\(.*\),\(.*\);/var_\1 = log (var_\2);/g" instr.tmp
cat instr.tmp >> ${file}_sub_IA.g

# handle outputs
echo "{" >> ${file}_sub_IA.g
cat $file.asm.bak | grep "LD," | sed "s/LD,\(.*\),\(.*\),\(.*\);/var_\1 in \[ \2,\3\] -> /g" >> ${file}_sub_IA.g

# count #store_instr to fit gappa syntax 
st_cnt=`cat $file.asm.bak | grep "ST," | wc -l`
st_cnt=`expr $st_cnt - 1`
if [ $st_cnt -gt 0 ]
then
    cat $file.asm.bak | grep "ST," | sed "1,${st_cnt}s/ST,\(.*\),\(.*\),\(.*\);/var_\2 in ? \/\\\\/" >> ${file}_sub_IA.g
    cat $file.asm.bak | grep "ST," | sed -i "s/ST,\(.*\),\(.*\),\(.*\);/var_\2 in ? /" ${file}_sub_IA.g
else
    cat $file.asm.bak | grep "ST," | sed "s/ST,\(.*\),\(.*\),\(.*\);/var_\2 in ? /g" >> ${file}_sub_IA.g
fi
echo "}" >> ${file}_sub_IA.g

cat $file.asm.bak | grep "LD," | sed "s/LD,\(.*\),\(.*\),\(.*\);/$ var_\1 in $2 ; /g" >> ${file}_sub_IA.g

popd &> /dev/null
