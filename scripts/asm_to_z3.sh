#!/bin/zsh -i

bits=20
source ~/.zshrc

touch dummy.svg dummy.gimple dummy.asm dummy.bak dummy.o dummy.tmp dummy.smt2
rm -f *.svg 2&> /dev/null
rm -f *.gimple* 2&> /dev/null
rm -f *.asm* 2&> /dev/null
rm -f *.bak* 2&> /dev/null
rm -f *.o 2&> /dev/null

file=`basename $1 .c`
dir=`dirname $1`

pushd ../data/$file &> /dev/null
touch dummy.smt2 dummy.tmp
rm -f *.smt2 2&> /dev/null  
rm -f *.tmp 2&> /dev/null

echo "(declare-const bound Real)" >> $file.smt2
err=`echo "2^-$bits" | bc -l | awk '{printf "%f", $0}'`
echo "(assert (= bound $err))" >> $file.smt2

# handle inputs
touch $file.smt2
cat $file.asm.bak | grep "LD," | sed "s/LD,\(.*\),\(.*\),\(.*\);/(declare-const var_\1 Real)/" >> $file.smt2
cat $file.asm.bak | grep "LD," | sed "s/LD,\(.*\),\(.*\),\(.*\);/(declare-const varfx_\1 Real)/" >> $file.smt2
cat $file.asm.bak | grep "ADD\|SUB\|MUL\|DIV\|EXP\|LOG" | sed "s/.*,\(.*\),\(.*\),\(.*\);/(declare-const var_\1 Real)/" >> $file.smt2
cat $file.asm.bak | grep "ADD\|SUB\|MUL\|DIV\|EXP\|LOG" | sed "s/.*,\(.*\),\(.*\),\(.*\);/(declare-const varfx_\1 Real)/" >> $file.smt2
cat $file.asm.bak | grep "ADD\|SUB\|MUL\|DIV\|EXP\|LOG" | sed "s/.*,\(.*\),\(.*\),\(.*\);/(declare-const d_\1 Real)/" >> $file.smt2
cat $file.asm.bak | grep "ADD\|SUB\|MUL\|DIV\|EXP\|LOG" | sed "s/.*,\(.*\),\(.*\),\(.*\);/(assert (>= bound d_\1))/" >> $file.smt2
cat $file.asm.bak | grep "ADD\|SUB\|MUL\|DIV\|EXP\|LOG" | sed "s/.*,\(.*\),\(.*\),\(.*\);/(assert (>= d_\1 (- bound)))/" >> $file.smt2

echo "(declare-const result Real)" >> $file.smt2
echo "(declare-const result1 Real)" >> $file.smt2

# handle compute 
touch instr.tmp
cat $file.asm.bak | grep "ADD\|SUB\|MUL\|DIV\|EXP\|LOG" > instr.tmp
sed -i "s/ADD,\(.*\),\(.*\),\(.*\);/(assert (= (+ var_\2 var_\3) var_\1))/g" instr.tmp
sed -i "s/SUB,\(.*\),\(.*\),\(.*\);/(assert (= (- var_\2 var_\3) var_\1))/g" instr.tmp
sed -i "s/MUL,\(.*\),\(.*\),\(.*\);/(assert (= (* var_\2 var_\3) var_\1))/g" instr.tmp
sed -i "s/DIV,\(.*\),\(.*\),\(.*\);/(assert (= (\/ var_\2 var_\3) var_\1))/g" instr.tmp
sed -i "s/EXP,\(.*\),\(.*\),\(.*\);/(assert (= (exp var_\2) var_\1))/g" instr.tmp
sed -i "s/LOG,\(.*\),\(.*\),\(.*\);/(assert (= (log var_\2) var_\1))/g" instr.tmp
cat instr.tmp >> $file.smt2
cat $file.asm.bak | grep "ADD\|SUB\|MUL\|DIV\|EXP\|LOG" > instr.tmp
sed -i "s/ADD,\(.*\),\(.*\),\(.*\);/(assert (= (+ varfx_\2 varfx_\3) varfx_\1))/g" instr.tmp
sed -i "s/SUB,\(.*\),\(.*\),\(.*\);/(assert (= (- varfx_\2 varfx_\3) varfx_\1))/g" instr.tmp
sed -i "s/MUL,\(.*\),\(.*\),\(.*\);/(assert (= (* varfx_\2 varfx_\3) varfx_\1))/g" instr.tmp
sed -i "s/DIV,\(.*\),\(.*\),\(.*\);/(assert (= (\/ varfx_\2 varfx_\3) varfx_\1))/g" instr.tmp
sed -i "s/EXP,\(.*\),\(.*\),\(.*\);/(assert (= (exp varfx_\2) varfx_\1))/g" instr.tmp
sed -i "s/LOG,\(.*\),\(.*\),\(.*\);/(assert (= (log varfx_\2) varfx_\1))/g" instr.tmp
cat instr.tmp >> $file.smt2

# handle input ranges 
cat $file.asm.bak | grep "LD," | sed "s/LD,\(.*\),\(.*\),\(.*\);/(assert (>= var_\1 \2))/g" >> $file.smt2
cat $file.asm.bak | grep "LD," | sed "s/LD,\(.*\),\(.*\),\(.*\);/(assert (>= \3 var_\1))/g" >> $file.smt2
cat $file.asm.bak | grep "LD," | sed "s/LD,\(.*\),\(.*\),\(.*\);/(assert (>= varfx_\1 \2))/g" >> $file.smt2
cat $file.asm.bak | grep "LD," | sed "s/LD,\(.*\),\(.*\),\(.*\);/(assert (>= \3 varfx_\1))/g" >> $file.smt2

# count #store_instr to fit gappa syntax 
var_cnt=`cat $file.asm.bak | grep "LD\|ADD\|SUB\|MUL\|DIV\|EXP\|LOG" | wc -l | sed "s/\ //g"`
var_cnt=`expr $var_cnt - 1`

echo "(assert (= (- var_$var_cnt varfx_$var_cnt) result1))" >> $file.smt2
echo "(assert (= (* result1 result1) result))" >> $file.smt2
echo "(assert (> result 0.001))" >> $file.smt2
echo "(check-sat)" >> $file.smt2
echo "(get-model)" >> $file.smt2

popd &> /dev/null
