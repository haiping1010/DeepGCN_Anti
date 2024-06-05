
cat  *_sorted_out.txt  > all_out.list

sed -i 's/  / /g' all_out.list
sort -g -rk  3,3 all_out.list  > all_out.sort

##first
##awk -F ' '  '$3 >=0.998' all_out.sort  >  all_out_select.sort

awk -F ' '  '$3 >=0.7' all_out.sort  >  all_out_select.sort
