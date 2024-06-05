rm summary.txt
for name in   new_antibody_*_f.pdb  xxxx_antibody.pdb
do

python caculate_interaction.py $name >>summary.txt

done

