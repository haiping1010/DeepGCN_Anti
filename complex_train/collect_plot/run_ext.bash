
cd ../
for dir in all_pdb*
do

	cd $dir
	echo $dir
	python  extract_performance_VS_epoch.py > output_result.txt

cd  ..
done
