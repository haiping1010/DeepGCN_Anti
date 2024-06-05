
fr=open('all-query.txt','r')

arr=fr.readlines()
index=0
name_list=['H1','H2','H3','L1','L2','L3']
for line in arr:
    fw=open(name_list[index]+'_query.fasta', 'w')
    fw.write('>'+name_list[index]+'\n')
    fw.write(line)
    index=index+1

fw.close()
