
conda activate  ABlooper

for name in *_f.pdb
do

	base=${name}

	ABlooper   $name  --output  $base'_n.pdb' --heavy_chain H --light_chain L

done
