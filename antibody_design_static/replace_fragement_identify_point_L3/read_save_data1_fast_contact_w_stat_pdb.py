import numpy as np
import pandas as pd
import sys, os
from random import shuffle

#from sklearn import metrics

import random

from random import choice

dict_pair_energy=np.load('dict_pair_energy.npy', allow_pickle=True).item()
aa_codes = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']




aa3to1={
   'ALA':'A', 'VAL':'V', 'PHE':'F', 'PRO':'P', 'MET':'M',
   'ILE':'I', 'LEU':'L', 'ASP':'D', 'GLU':'E', 'LYS':'K',
   'ARG':'R', 'SER':'S', 'THR':'T', 'TYR':'Y', 'HIS':'H',
   'CYS':'C', 'ASN':'N', 'GLN':'Q', 'TRP':'W', 'GLY':'G'
}

aa1to3={value:key  for key,value  in aa3to1.items()}



def  caculate_pair(HLposition,Eposition,ResinameHL,ResinameE):
    residuePair=[]
    for key1, value1 in HLposition.items():
       #print (ResinameHL[key1], 'corresponds to', value1)
       for key2, value2 in Eposition.items():
            #print key2,value2
            #print (ResinameE[key2], 'corresponds to', value2)
            ##distance=pow(value1[0]-value2[0])
            a = np.array(value1)
            a1 = a.astype(np.float)
            b = np.array(value2)
            b1 = b.astype(np.float)
            xx=np.subtract(a1,b1)
            tem=np.square(xx)
            tem1=np.sum(tem)
            out=np.sqrt(tem1)
            #print a
            if out<20 :
                #print (ResinameHL[key1],ResinameHL[key2])
                #print (a,b,out)
                #print (a1)
                residuePair.append([ResinameHL[key1],ResinameE[key2],out])
                #print (residuePair)


    return   residuePair






def readpdb_chain(filename, chains):
   cord = [None] * 3
   Pposition={}
   ResinameP={}
   Atomline=[]
   fr=open(filename,'r')
   arr_all=fr.readlines()
   fr.close()

   for line in arr_all:
   #for line in open(filebase+'_antibody.pdb'):
      tem_B=' '
      if len(line)>16:
         tem_B=line[16]
         line=line[:16]+' '+line[17:]

      #print(line)
      list_n = line.split()
      id = list_n[0]
      if id == 'ATOM' and tem_B !='B':
         type = list_n[2]
         chain_id=line[21:22]
         #if type == 'CA' and list[3]!= 'UNK':
         if  type == 'CA' and  list_n[3]!= 'UNK' and  chain_id in chains :
             residue = list_n[3]
             type_of_chain = line[21:22]
             tem1=line[22:26].replace("A", "")
             tem2=tem1.replace("B", "")
             tem2=tem2.replace("C", "")

             #tem2=filter(str.isdigit, list[5])
             atom_count = line[4:11]+tem2+line[21:22]
             list_nx=[0]*3
             list_nx[0]=line[30:38]
             list_nx[1]=line[38:46]
             #print line
             list_nx[2]=line[46:54]
             position = list_nx[0:3]

             Pposition[atom_count]=position
             list_n[4]=line[21:22]
             list_n[5]=line[22:26].replace(' ','')
             ResinameP[atom_count]=residue+list_n[5]+list_n[4]
             resindex=residue+list_n[5]
             #Atomline.append(line)
             Atomline.append(line)
             Pposition[atom_count]=position
   return   Pposition,ResinameP,Atomline

def readpdb_chain_int(filename, chains ,cord_inter_sel_res):
   cord = [None] * 3
   Pposition={}
   ResinameP={}
   Atomline=[]
   cutoff=np.square(25.0)
   fr=open(filename,'r')
   arr_all=fr.readlines()
   fr.close()
   for line in arr_all:

   #for line in open(filename):
   #for line in open(filebase+'_antibody.pdb'):
      tem_B=' '
      if len(line)>16:
         tem_B=line[16]
         line=line[:16]+' '+line[17:]
      #print(line)
      ##we add this mark things, because in very few cases, the cross docking chain are same
      list_n = line.split()
      id = list_n[0]
      if id == 'ATOM' and tem_B !='B':
         type = list_n[2]
         chain_id=line[21:22]
         #if type == 'CA' and list[3]!= 'UNK':
         if  type == 'CA' and   list_n[3]!= 'UNK' and  chain_id in chains  :
             residue = list_n[3]
             type_of_chain = line[21:22]
             tem1=line[22:26].replace("A", "")
             tem2=tem1.replace("B", "")
             tem2=tem2.replace("C", "")

             #tem2=filter(str.isdigit, list[5])
             atom_count = line[4:11]+tem2+line[21:22]
             list_nx=[0]*3
             list_nx[0]=line[30:38]
             list_nx[1]=line[38:46]
             #print line
             list_nx[2]=line[46:54]
             position = list_nx[0:3]

             ###Pposition[atom_count]=position
             list_n[4]=line[21:22]
             list_n[5]=line[22:26].replace(' ','')

             a = np.array(position)
             a1 = a.astype(np.float)
             b = np.array(cord_inter_sel_res)
             b1 = b.astype(np.float)
             xx=np.subtract(a1,b1)
             tem=np.square(xx)
             tem1=np.sum(tem)
             if tem1<cutoff:
                 Pposition[atom_count]=position
                 ResinameP[atom_count]=residue+list_n[5]+list_n[4]
                 resindex=residue+list_n[5]
                 #Atomline.append(line)
                 Atomline.append(line)
                 Pposition[atom_count]=position
   return   Pposition,ResinameP,Atomline






#import glob
#file_arr=glob.glob("*_*.pdb")

def  findcontact(avg_n,Eposition):
    #print  Eposition
    cord_inter_sel_res=10000000
    smallest=10000000
    #print (avg_n)
    average_v=avg_n
    #print np.array(avg).astype(np.float)
    #for i in range(10):
    #    print  avg_n
    for  value in Eposition.values():
            a = np.array(value)
            a1 = a.astype(np.float)
            #print a1
            b = np.array(average_v)
            #print avg1
            b1 = b.astype(np.float)
            xx=np.subtract(a1,b1)
            #print xx
            tem=np.square(xx)
            tem1=np.sum(tem)
            #print tem1
            #print (tem1)
            #out=np.sqrt(tem1)
            if tem1<smallest:
                smallest=tem1
                cord_inter_sel_res=a1
    return   cord_inter_sel_res 



import sys

PDBs=sys.argv[1]
temT_f=sys.argv[2]

frr=open(PDBs+'/'+temT_f,'r')
arr_frr=frr.readlines()

####
#add label

        


####



file_arr=[]
frr.close()
for name in arr_frr:
     file_arr.append(name.strip())
dic_contact_m_vec={}
dic_contact_ori_vec={}

#4c9e_1_C_4c9e_1_D_complex.5_n.pdb
for filename  in file_arr:
    dict_mut={}
    #print filename
    #tem_arr=filename.split("_")
    ####obtain the mutation residue
    #5XCO_A_B_DA13G-PB6A.pdb
    HLposition, ResinameHL, Atomline=readpdb_chain(PDBs+'/'+filename,list('HL'))
    cord=list(HLposition.values())
    cord=np.array(cord).astype(float)
    avg=np.average(cord,axis=0)
    #print cord
    #print avg


    #print  (avg)
    #print (cord)
    print ("finished stage1")
    Eposition, ResinameE,AtomlineE=readpdb_chain(PDBs+'/'+filename, list('A'))
    cord2=list(Eposition.values())
    cord2=np.array(cord2).astype(float)
    avg2=np.average(cord2,axis=0)
    #print cord2

    cord_inter_sel_res=findcontact(avg,Eposition)
    cord_inter_sel_res2=findcontact(avg2,HLposition)
    #print (len(Eposition.values()))
    Contact_cord=np.vstack((cord_inter_sel_res,cord_inter_sel_res2))

    Contact_cord=np.average(Contact_cord,axis=0)
    #print (cord_inter_sel_res)
    
    #####find pocket in potential region
    HLposition, ResinameHL, Atomline=readpdb_chain_int(PDBs+'/'+filename,list('HL'), Contact_cord)

    Eposition, ResinameE,AtomlineE=readpdb_chain_int(PDBs+'/'+filename, list('A'), Contact_cord)

    #######


    print ("finished stage2")
    print (filename)

    print ("finished stage3")

    residuePair=caculate_pair(HLposition,Eposition,ResinameHL,ResinameE)
    dict_pair_count_vdw={}
    dict_pair_count_ele={}
    for aa1 in aa_codes:
        for aa2 in aa_codes:
               dict_pair_count_vdw[aa1+'_'+aa2]=0
               dict_pair_count_ele[aa1+'_'+aa2]=0


    ##### contact vector
    for pair in residuePair:
        if pair[0][0:3] in aa_codes and pair[1][0:3] in aa_codes:
           key=pair[0][0:3]+'_'+pair[1][0:3]
           dict_pair_count_vdw[key] = dict_pair_count_vdw[key] + 1/(pow(float(pair[2]),6))
           dict_pair_count_ele[key] = dict_pair_count_ele[key] + 1/(float(pair[2]))
    vec_line=[]
    for aa1 in aa_codes:
        for aa2 in aa_codes:
             vec_line.append( float(dict_pair_count_vdw[aa1+'_'+aa2])*float(dict_pair_energy[aa1+'_'+aa2]))
    for aa1 in aa_codes:
        for aa2 in aa_codes:
             vec_line.append(float(dict_pair_count_ele[aa1+'_'+aa2])*float(dict_pair_energy[aa1+'_'+aa2]))

    dic_contact_m_vec[filename]=vec_line





    #print  (residuePair)
import numpy as np

np.save(PDBs+'_'+temT_f.replace('.txt','')+'_dic_contact_vec_e.npy',dic_contact_m_vec)

#np.save('dic_label.npy',dict_label)
