
cd ../
for dir in   all_pdb_affinity_onhotxx_cutoff0.8
do

	cd $dir
	echo $dir
	python  extract_performance_VS_epoch.py > output_result.txt

cd  ..
done
