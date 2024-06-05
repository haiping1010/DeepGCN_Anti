
for name in *.pdb
do

	base=${name%.pdb}

pdbfixer  $base'.pdb'  --add-atoms=heavy --output=$base'_f.pdb'



done
