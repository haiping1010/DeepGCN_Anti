cat   list.txt  | while read line
do

#IFS=' ' read -r -a array <<< $line
##wget 'http://zinc15.docking.org/substances/'${array[0]}'.sdf'

cp  -r  ../final_ana/f_collect/$line*.pdb .

cp -r   /home/zhanghaiping/program/antibody_design_deep2_redo_mutation_less/select_prediction/collect/final_ana/f_collect/$line*.pdb  .

done
