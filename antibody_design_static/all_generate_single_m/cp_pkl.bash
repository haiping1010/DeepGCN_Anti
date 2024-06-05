
cd ../
for name in replace_fragement_identify_point_??
do
base=${name:18 }

echo $base
cp  -r  $name/possible_mutations.pkl    all_generate_single_m/$base'_possible_mutations.pkl'


done
