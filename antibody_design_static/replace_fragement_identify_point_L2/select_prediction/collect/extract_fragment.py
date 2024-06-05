


#range_H='56-67'
#range_L='105-117'




'''
range_H='105-117'

range_L='105-117'
range_H_add='56-67'



51-60  H CDR2
97-114  H CDR3
87-97  L  CDR3

'''

import sys
import os

input_f=sys.argv[1]

fr=open(input_f,'r')

fw_H2=open(input_f.replace('.fasta','_Hcdr2.fa'),'w')
fw_H3=open(input_f.replace('.fasta','_Hcdr3.fa'),'w')


fw_L=open(input_f.replace('.fasta','_Lcdr3.fa'),'w')


arr_all=fr.readlines()

oldline=''
for  line in arr_all:
    #print oldline
    if oldline.startswith('>') and  oldline.strip().endswith(':H'):
        print line[51:61]
        fw_H2.write(line[51:61])
        print line[97:115]
        fw_H3.write(line[97:115])

    elif  oldline.startswith('>') and  oldline.strip().endswith(':L'):
        print line[87:98]
        fw_L.write(line[87:98])



    oldline=line
fw_H2.close()
fw_H3.close()
fw_L.close()

