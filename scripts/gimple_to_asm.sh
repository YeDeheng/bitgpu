#!/bin/zsh -i

touch dummy.gimple
rm -f *.gimple

file=`basename $1 .c`
dir=`dirname $1`
rm -Rf ../data/$file
mkdir -p ../data/$file
cp $dir/$file.c ../data/$file
cp $dir/$file.range ../data/$file

pushd ../data/$file &> /dev/null
gcc -fdump-tree-gimple -c $file.c
2=$file.c.004t
cp $2.gimple $2.gimple.bak

# grep NOT using grep -v, added by Deheng
inputs=`cat $2.gimple | grep "(.*).*" |grep -v "(.*).*;" | sed "s/.*(\(.*\)).*/\1/" | cut -d"," -f1-100 | sed "s/double//g" | sed "s/\ //g" | sed "s/*.*//g" | sed "s/,/\ /g"`
constants=`cat $2.gimple | grep -v "double [a-z|A-Z|0-9].*;" | sed "s/.* \([0-9]*\.[0-9]*e[+\|-][0-9]*\).*/printf 'FILTER\1'/ge" | grep "FILTER" | sed "s/FILTER//g" | sed "s/*//g"`
userregs=`cat $2.gimple | grep ".*double [a-z|A-Z|0-9]*;.*" | tr "\n" "," | sed "s/double//g" | sed "s/;//g" | sed "s/,$/\n/" | sed "s/ //g" | tr "," " "`

minimum=`cat $2.gimple | grep "double D.*;" | sed "s/.*double D.\([0-9]*\);/\1/g" | mymin` #numbound -l`
maximum=`cat $2.gimple | grep "double D.*;" | sed "s/.*double D.\([0-9]*\);/\1/g" | mymax` #numbound`
io_count=`echo $inputs | wc -w`
const_count=`echo -n $constants | wc -l`
echo " const_count : $const_count "
userreg_count=`cat $2.gimple | grep ".*double [a-z|A-Z|0-9]*;.*" | wc -l`

# support instructions
sed -i "s/\(.*\)=\(.*\) + \(.*\)/ADD,\1,\2,\3/" $2.gimple
sed -i "s/\(.*\)=\(.*\) \* \(.*\)/MUL,\1,\2,\3/" $2.gimple
sed -i "s/\(.*\)=\(.*\) - \(.*\)/SUB,\1,\2,\3/" $2.gimple
sed -i "s/\(.*\)=\(.*\) \/ \(.*\)/DIV,\1,\2,\3/" $2.gimple
sed -i "s/\(.*\)= exp (\(.*\))/EXP,\1,\2,-1/" $2.gimple
sed -i "s/\(.*\)= log (\(.*\))/LOG,\1,\2,-1/" $2.gimple
sed -i "s/\(.*\)= sqrt (\(.*\))/SQRT,\1,\2,-1/" $2.gimple

# process outputs
sed -i "s/.*return\(.*\);/ST,0,\1,-1;/" $2.gimple 
sed -i "s/*.*=\(.*\);/ST,\1,\1,-1;/" $2.gimple 

# cleanups
sed -i "/.*double [a-z|A-Z|0-9].*;/d" $2.gimple
sed -i "/.*[{\|}]/d" $2.gimple
sed -i "/^$/d" $2.gimple
sed -i 1d $2.gimple


# handle intermediate register substitition (3 passes for 3-level IR)
sed -i "s/\(.*\)D.\([0-9][0-9][0-9][0-9]\)\(.*\)/printf '\1';echo '\2 -  $minimum + $io_count + $const_count + $userreg_count + 1' \| bc \| tr '\n' ' '; printf '\3'/e" $2.gimple 
sed -i "s/\(.*\)D.\([0-9][0-9][0-9][0-9]\)\(.*\)/printf '\1';echo '\2 -  $minimum + $io_count + $const_count + $userreg_count + 1' \| bc \| tr '\n' ' '; printf '\3'/e" $2.gimple
sed -i "s/\(.*\)D.\([0-9][0-9][0-9][0-9]\)\(.*\)/printf '\1';echo '\2 -  $minimum + $io_count + $const_count + $userreg_count + 1' \| bc \| tr '\n' ' '; printf '\3'/e" $2.gimple

# handle temp variable, added by Deheng, NOTE: this is hacky, have to be of certain format. 
sed -i "s/\(.*\)T\([0-9]\)\(.*\)/printf '\1';echo '$io_count + $const_count + $userreg_count - \2' \| bc \| tr '\n' ' '; printf '\3'/e" $2.gimple
sed -i "s/\(.*\)T\([0-9]\)\(.*\)/printf '\1';echo '$io_count + $const_count + $userreg_count - \2' \| bc \| tr '\n' ' '; printf '\3'/e" $2.gimple
sed -i "s/\(.*\)T\([0-9]\)\(.*\)/printf '\1';echo '$io_count + $const_count + $userreg_count - \2' \| bc \| tr '\n' ' '; printf '\3'/e" $2.gimple


cntr=0;
for io in `echo $inputs`
do
	lo=`cat $file.range | grep $io | cut -d"," -f2`
	hi=`cat $file.range | grep $io | cut -d"," -f3`
	echo $lo0 $lo $hi0 $hi
	echo "LD,$cntr,$lo,$hi;" >> inputs.gimple
	sed -i "s/$io/$cntr/g" $2.gimple
	cntr=`echo $cntr + 1 | bc`
done

touch constants.gimple
for const in `echo $constants`
do
	matched=`cat $2.gimple | grep -c "$const"`
	if [[ $matched -gt 0 ]];
	then
		echo "LD,$cntr,$const, $const;" >> constants.gimple
		sed -i "s/\<$const\>/$cntr/g" $2.gimple
		cntr=`echo $cntr + 1 | bc`
	fi
done

# process user registers
touch userregs.gimple
for userreg in `echo $userregs`
do
	matched=`cat $2.gimple | grep -c "$userreg = [0-9]*\.[0-9]*e[+\|-][0-9]*;"`
	val=`cat $2.gimple | grep "$userreg = [0-9]*\.[0-9]*e[+\|-][0-9]*;" | sed "s/$userreg = \([0-9]*\.[0-9]*e[+\|-][0-9]*\);/\1/"`
	# delete offending line from file
	sed -i "/$userreg = [0-9]*\.[0-9]*e[+\|-][0-9]*;/d" $2.gimple
	if [[ $matched -gt 0 ]];
	then
		echo "LD,$cntr,$val,$val;" >> userregs.gimple
		# this is dangerous as we have to ensure variable names dont clash in regular expression matches!
		sed -i "s/$userreg/$cntr/g" $2.gimple
		cntr=`echo $cntr + 1 | bc`
	fi
done

cat inputs.gimple > $file.asm
cat constants.gimple >> $file.asm
cat userregs.gimple >> $file.asm
cat $2.gimple >> $file.asm
sed -i "s/\ *//g" $file.asm
cp $file.asm $file.asm.bak 

# convert OPCODES to low-level machine code
while read line
do
	opcode=`echo $line | grep define | cut -d' ' -f 2`
	intcode=`echo $line | grep define | cut -d' ' -f 3`
	sed -i "s/$opcode/$intcode/g" $file.asm
done < ../../include/opcode.h

popd &> /dev/null
