
cat   contactDefinition_nn_restxx.txt  | while read line

do


IFS='	' read -r -a array <<< $line


echo ${array[0]}  ${array[1]} ${array[2]}

nohup python  python_2_pp_sep_d_sep.py   ${array[0]}  ${array[1]} ${array[2]}  &

sleep 0.3s



done
