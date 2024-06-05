#conda activate py3
for name in   temT_51.txt
#for name in   all_data/temT_42.txt
#for name in   all_data/temT_100.txt  all_data/temT_101.txt
do

	id=${name:5:-4}
	echo $id
nohup python  read_smi_protein_nnn.py  $id  > 'outT_'$id'.log' 2>&1&

done
#conda activate py3
for name in   tem_51.txt
#for name in   all_data/temT_42.txt
#for name in   all_data/temT_100.txt  all_data/temT_101.txt
do

	id=${name:4:-4}
	echo $id
nohup python  read_smi_protein_neg_nnn.py  $id  > 'out_'$id'.log' 2>&1&

done
