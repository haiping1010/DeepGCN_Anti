
for name in ??_query.fasta
do

base=${name%.fasta}
basex=${name%_query.fasta}
blastp -query $name -db all_CDR_seq_uniq.fasta  -out  $base'_out.txt'

grep  '^> id_\|^Query \|^Sbjct'  $base'_out.txt'  > 'replace_fragement_identify_point_'$basex/output_file_f.txt





done
