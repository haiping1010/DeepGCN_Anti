
rm  select.txt
for i in {1..9};do  ls  $i*'_poc.pdb' >> select.txt  ; echo $i; done

awk -F "_"  '{print $1"_"$2}'  select.txt | sort | uniq > select_uniq.txt

nohup python compare.py  all_select_uniq.txt_old  select_uniq.txt > all_select_uniq.txt  2>&1&
