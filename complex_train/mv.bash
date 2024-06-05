#cat    final_pdbid_rest.txt  | while read  line

mkdir problem_empty
cat   problem_empty.txt  | while read line

do


#base=${line:1:4}
#base=${line:0:4}

mv $line  problem_empty/


done

