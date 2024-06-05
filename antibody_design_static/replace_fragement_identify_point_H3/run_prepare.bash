for  folder in     temT_*.txt
do

nohup python     read_save_data1_fast_contact_w_stat_pdb.py   all_pdbs    $folder > $folder'_out.log' 2>&1&

done
