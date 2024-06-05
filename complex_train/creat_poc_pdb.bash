
for i in {1..9}
do
for name in $i???_?_?_poc.pdb

do
base=${name%_poc.pdb}
	echo $name

	cp  -r  $name   ../../3Dcomplex_cutoff8/complex/$base'_w.pdb'


done



done
