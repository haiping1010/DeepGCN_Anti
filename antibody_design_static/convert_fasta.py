

fr=open('all_CDR_seq_uniq.txt','r')
fw=open('all_CDR_seq_uniq.fasta','w')
arr_tem=fr.readlines()
for  idx in range(len(arr_tem)):
     fw.write('>id_'+str(idx).ljust(4,'0')+"\n")
     fw.write(arr_tem[idx].strip()+"\n")
fw.close()
