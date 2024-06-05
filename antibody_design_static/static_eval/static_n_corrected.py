import math
import numpy as np


#####RT=2.479
seq=("ALA","ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")

all=open("summary.txt",'r')

mapcount={}
rescount={}


for res1 in  seq:
    for res2 in seq:
        mapcount[res1+'_'+res2]=0

for res in  seq:
     rescount[res] = 0




for name in all:
    res=name.split(':::')
    
    res_arr=res[1].strip().split(',')
    for res_name in res_arr:
        if res_name in seq and res[0] in seq:
           if   res_name == res[0] :
               mapcount[res[0]+'_'+res_name]=mapcount[res[0]+'_'+res_name]+1
               rescount[res[0]] = rescount[res[0]] +1
               rescount[res_name] = rescount[res_name] +1
           elif  res_name  != res[0] :
               mapcount[res[0]+'_'+res_name]=mapcount[res[0]+'_'+res_name]+1
               mapcount[res_name+'_'+res[0]]=mapcount[res_name+'_'+res[0]]+1
               rescount[res[0]] = rescount[res[0]] +1
               rescount[res_name] = rescount[res_name] +1
                   

all_count=0
for res in  seq:
    all_count=all_count+rescount[res]

all_mapcount=0
for  i  in  range(len(seq)):
    for  j  in range(len(seq)):
        if j >= i:
           res1=seq[i]
           res2=seq[j]
           all_mapcount=all_mapcount+mapcount[res1+'_'+res2]

print all_count
dict_energy={}
for res1 in  seq:
    for res2 in seq:
        if    res1 == res2:
           part1=(float(mapcount[res1+'_'+res2]))/(all_mapcount*2)
           part2=float(rescount[res1]*rescount[res2])/(all_count*all_count)
        elif  res1 != res2:
           part1=float(mapcount[res1+'_'+res2])/(all_mapcount*2)
           part2=float(rescount[res1]*rescount[res2])/(all_count*all_count)
        print (part1, part2)

        if part1!=0:
             energy=-math.log(part1/part2)*2.479
        else :
            energy=5*2.479
        print res1,res2,energy
        dict_energy[res1+'_'+res2]=energy
        #print float(mapcount[res1+'_'+res2])/(float(rescount[res1]*rescount[res2])/(all_count*all_count))
         #print  (float(rescount[res1]*rescount[res2])/(all_count*all_count))

np.save('dict_energy.npy', dict_energy)

dict_load=np.load('dict_energy.npy', allow_pickle=True)







