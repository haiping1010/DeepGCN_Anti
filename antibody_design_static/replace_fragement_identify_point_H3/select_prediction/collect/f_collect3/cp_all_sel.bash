cat   all_out_select_f.sort  | while read line
do

IFS=' ' read -r -a array <<< $line
##wget 'http://zinc15.docking.org/substances/'${array[0]}'.sdf'

cp -r  ../../../all_pdbs/${array[1]}  .

done
