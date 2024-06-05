
for  name in  design*complex_new_antibody_*
do

base=${name%.pdb}

python  pdb2fasta.py  $name >  $base'.fasta'



done
