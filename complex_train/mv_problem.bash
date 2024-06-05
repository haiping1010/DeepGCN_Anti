mkdir problem_fold
cat   problem_n.txt  | while read line
do

IFS=' ' read -r -a array <<< $line
##wget 'http://zinc15.docking.org/substances/'${array[0]}'.sdf'

mv   *${line:0:4}*  problem_fold


done
