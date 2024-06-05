#conda activate py3
for name in   temT_*.txt
#for name in   all_data/temT_42.txt
#for name in   all_data/temT_100.txt  all_data/temT_101.txt
do

	id=${name:5:-4}
	echo $id
python  read_smi_protein_nnn.py  $id  


done
