
cd ../

#for name in  all_pdb_*
for name in  all_pdb_affinity_onhotxx_cutoff0.8
do


#cp -r $name/plot/hbond.png   collect_plot/$name'.png'

#cp -r  $name/plot/output_result.txt   collect_plot/$name'_output_result.txt'
cp -r $name/output_result.txt   collect_plot/$name'output_result.txt'

done
