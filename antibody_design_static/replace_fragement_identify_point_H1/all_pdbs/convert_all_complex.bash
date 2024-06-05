
for name in   new_antibody_*.pdb
do

	base=${name%.pdb}

	cat $name > complex_$name
	echo "TER\n" >> complex_$name
	cat  xxxx_antigen.pdb  >> complex_$name


done

