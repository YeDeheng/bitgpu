#!/bin/zsh -i

# This script will translate the ASM format into a syntax suitable for Gappa evaluation
# The code is designed to run simple interval analysis *without* sub-interval splits 
# This can be converted into a compiler optimiztion to make it more robust

bits=20
source ~/.zshrc

touch dummy.svg dummy.gimple dummy.asm dummy.bak dummy.o dummy.tmp dummy.g
rm -f *.svg 2&> /dev/null
rm -f *.gimple* 2&> /dev/null
rm -f *.asm* 2&> /dev/null
rm -f *.bak* 2&> /dev/null
rm -f *.o 2&> /dev/null

file=`basename $1 .c`
dir=`dirname $1`

pushd ../data/$file &> /dev/null
touch dummy.g dummy.tmp
rm -f *.g 2&> /dev/null  
rm -f *.tmp 2&> /dev/null

# handle inputs
touch $file.g
cat $file.asm.bak | grep "LD," | sed "s/LD,\(.*\),\(.*\),\(.*\);/@fx\1 = fixed<-$bits,ne>;/" >> $file.g
cat $file.asm.bak | grep "ADD\|SUB\|MUL\|DIV\|EXP\|LOG" | sed "s/.*,\(.*\),\(.*\),\(.*\);/@fx\1 = fixed<-$bits,ne>;/" >> $file.g

cat $file.asm.bak | grep "LD," | sed "s/LD,\(.*\),\(.*\),\(.*\);/var_\1_fx = fx\1(var_\1);/" >> $file.g 

# handle compute 
touch instr.tmp
cat $file.asm.bak | grep "ADD\|SUB\|MUL\|DIV\|EXP\|LOG" > instr.tmp
sed -i "s/ADD,\(.*\),\(.*\),\(.*\);/var_\1 = var_\2 + var_\3;/g" instr.tmp
sed -i "s/SUB,\(.*\),\(.*\),\(.*\);/var_\1 = var_\2 - var_\3;/g" instr.tmp
sed -i "s/MUL,\(.*\),\(.*\),\(.*\);/var_\1 = var_\2 * var_\3;/g" instr.tmp
sed -i "s/DIV,\(.*\),\(.*\),\(.*\);/var_\1 = var_\2 \/ var_\3;/g" instr.tmp
sed -i "s/EXP,\(.*\),\(.*\),\(.*\);/var_\1 = exp (var_\2);/g" instr.tmp
sed -i "s/LOG,\(.*\),\(.*\),\(.*\);/var_\1 = log (var_\2);/g" instr.tmp
cat instr.tmp >> $file.g
cat $file.asm.bak | grep "ADD\|SUB\|MUL\|DIV\|EXP\|LOG" > instr.tmp
sed -i "s/ADD,\(.*\),\(.*\),\(.*\);/var_\1_fx = fx\1 (var_\2_fx + var_\3_fx);/g" instr.tmp
sed -i "s/SUB,\(.*\),\(.*\),\(.*\);/var_\1_fx = fx\1 (var_\2_fx - var_\3_fx);/g" instr.tmp
sed -i "s/MUL,\(.*\),\(.*\),\(.*\);/var_\1_fx = fx\1 (var_\2_fx * var_\3_fx);/g" instr.tmp
sed -i "s/DIV,\(.*\),\(.*\),\(.*\);/var_\1_fx = fx\1 (var_\2_fx \/ var_\3_fx);/g" instr.tmp
sed -i "s/EXP,\(.*\),\(.*\),\(.*\);/var_\1_fx = fx\1 ( exp(var_\2_fx) );/g" instr.tmp
sed -i "s/LOG,\(.*\),\(.*\),\(.*\);/var_\1_fx = fx\1 ( log(var_\2_fx) );/g" instr.tmp
cat instr.tmp >> $file.g


# handle outputs
echo "{" >> $file.g
cat $file.asm.bak | grep "LD," | sed "s/LD,\(.*\),\(.*\),\(.*\);/var_\1 in \[ \2,\3\] -> /g" >> $file.g

# count #store_instr to fit gappa syntax 
st_cnt=`cat $file.asm.bak | grep "ST," | wc -l`
st_cnt=`expr $st_cnt - 1`
cat $file.asm.bak | grep "ST," | sed "s/ST,\(.*\),\(.*\),\(.*\);/var_\2 in ? \/\\\\/g" >> $file.g
#cat $file.asm.bak | grep "ST," | sed "s/ST,\(.*\),\(.*\),\(.*\);/var_\2_dbl in ? \/\\\\/g" >> $file.g
#cat $file.asm.bak | grep "ST," | sed "s/ST,\(.*\),\(.*\),\(.*\);/(var_\2_dbl-var_\2) in ? \/\\\\/g" >> $file.g
#cat $file.asm.bak | grep "ST," | sed "s/ST,\(.*\),\(.*\),\(.*\);/(var_\2_fx-var_\2) in ? \/\\\\/g" >> $file.g
if [ $st_cnt -gt 0 ]
then
    cat $file.asm.bak | grep "ST," | sed "1,${st_cnt}s/ST,\(.*\),\(.*\),\(.*\);/(var_\2_fx-var_\2) in ? \/\\\\/" >> $file.g
    cat $file.asm.bak | grep "ST," | sed -i "s/ST,\(.*\),\(.*\),\(.*\);/(var_\2_fx-var_\2) in ? /" $file.g
else
    cat $file.asm.bak | grep "ST," | sed "s/ST,\(.*\),\(.*\),\(.*\);/(var_\2_fx-var_\2) in ? /g" >> $file.g
fi
echo "}" >> $file.g

popd &> /dev/null
