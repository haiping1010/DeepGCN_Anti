
import os
import sys


fr_n=open('XXXXA_XXXXHL_complex.1720_n_f.pdb_anti.fasta','r')

arr_n=fr_n.readlines()

list_n_H=[]
list_n_L=[]

oldline=''

for line in arr_n:
   if oldline.strip().endswith(':H')  and  oldline.startswith('>'):
       list_n_H=list(line.strip())
   if oldline.strip().endswith(':L')  and  oldline.startswith('>'):
       list_n_L=list(line.strip())

   oldline=line
       
def count_different_strings(array1, array2):
    count = 0
    arr_mut=[]
    
    # Ensure the arrays have the same length
    if len(array1) != len(array2):
        return None
    
    for i in range(len(array1)):
        if array1[i] != array2[i]:
            count += 1
            arr_mut.append(str(array1[i])+str(i)+str(array2[i]))

    
    return count,arr_mut



import glob

arr_all=glob.glob("design*complex_new_antibody*.pdb_anti.fasta")

fw=open('T_collection.txt','w')
#arr_all=glob.glob('complex_new_antibody_78559_f.pdb_anti.fasta')
oldline=''
for input_f in arr_all:
   #print input_f
   fr=open(input_f,'r')

   arr=fr.readlines()
   #print arr
   for line in arr:
     index_sum=0
     print oldline 
     if oldline.strip().endswith(':H')  and  oldline.startswith('>'):
        list_H=list(line.strip())
        print list_H,list_n_H
        count_H, arr_mut_L=count_different_strings(list_H,list_n_H)
     if oldline.strip().endswith(':L')  and  oldline.startswith('>'):
        list_L=list(line.strip())
        count_L, arr_mut_H=count_different_strings(list_L,list_n_L)
        print (input_f+"\t"+str(count_H)+"\t"+str(count_L)+"\t"+str(int(count_H)+int(count_L)))
        string_H = " ".join(arr_mut_H)
        string_L = " ".join(arr_mut_L)
        fw.write(input_f+"\t"+str(count_H)+"\t"+str(count_L)+"\t"+str(int(count_H)+int(count_L))+'\t'+string_H+'\t'+string_L+'\n')

     oldline=line

fw.close()
