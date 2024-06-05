cat    list_f.txt  | while read line
do

#IFS=' ' read -r -a array <<< $line
##wget 'http://zinc15.docking.org/substances/'${array[0]}'.sdf'

#cp -r    ../static_eval_cutoff10/${array[0]}  .

cp -r ../$line .
done
