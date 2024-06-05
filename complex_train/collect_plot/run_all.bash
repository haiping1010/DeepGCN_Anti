##for name in  *output_result.txt
for name in  all_pdb_affinity_onhotxx_cutoff0.8*output_result.txt

do

	base=${name%output_result.txt}

python   extract_part.py   $name  >  $base'_short.txt'




done
