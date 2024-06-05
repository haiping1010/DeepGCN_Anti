#cat    final_pdbid_rest.txt  | while read  line

mkdir problem_empty
cat  all_nonstardard.txt   | while read line

do


#base=${line:1:4}
base=${line:0:6}
echo $base
mv $base*  problem_empty/


done

