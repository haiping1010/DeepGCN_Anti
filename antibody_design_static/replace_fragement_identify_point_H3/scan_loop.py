import numpy as np
import pandas as pd
import sys, os
from random import shuffle
#from  read_smi_protein import *

from sklearn import metrics

import random

from random import choice

def readpdb(filename):
   Pposition={}
   ResinameP={}
   Atomline=[]
   for line in open(filename):
   #for line in open(filebase+'_antibody.pdb'):
      tem_B=' '
      if len(line)>16:
         tem_B=line[16]
         line=line[:16]+' '+line[17:]
      #print(line)
      list_n = line.split()
      id = list_n[0]
      if id == 'ATOM' and tem_B !='B':
         a_type = list_n[2]
         print (a_type)
         if a_type == 'CA' and list_n[3]!= 'UNK':
         #if   list_n[3]!= 'UNK' :
             print (line)
             residue = list_n[3]
             type_of_chain = line[21:22]
             tem1=line[22:26].replace("A", "")
             tem2=tem1.replace("B", "")
             tem2=tem2.replace("C", "")

             #tem2=filter(str.isdigit, list[5])
             atom_count = line[4:11]+tem2+line[21:22]
             list_n[6]=line[30:38]
             list_n[7]=line[38:46]
             list_n[8]=line[46:54]
             position = list_n[6:9]
             Pposition[atom_count]=position
             list_n[4]=line[21:22]
             list_n[5]=line[22:27].replace(' ','')
             ResinameP[atom_count]=residue+list_n[5]+list_n[4]
             resindex=residue+list_n[5]
             #Atomline.append(line)
             Atomline.append(line)
             Pposition[atom_count]=position
   return   Pposition,ResinameP,Atomline


def readpdb_allow_m(filename,allow_mutation):
   Pposition={}
   ResinameP={}
   Atomline=[]
   resid_to_name={}
   for line in open(filename):
   #for line in open(filebase+'_antibody.pdb'):
      tem_B=' '
      if len(line)>16:
         tem_B=line[16]
         line=line[:16]+' '+line[17:]
      #print(line)
      list_n = line.split()
      id = list_n[0]
      if id == 'ATOM' and tem_B !='B':
         a_type = list_n[2]
         if a_type == 'CA' and list_n[3]!= 'UNK':
           if   line[22:26].replace(' ','')+line[21:22]  in  allow_mutation :
             print (line)
             residue = list_n[3]
             type_of_chain = line[21:22]
             tem1=line[22:26].replace("A", "")
             tem2=tem1.replace("B", "")
             tem2=tem2.replace("C", "")

             #tem2=filter(str.isdigit, list[5])
             atom_count = line[4:11]+tem2+line[21:22]
             list_n[6]=line[30:38]
             list_n[7]=line[38:46]
             list_n[8]=line[46:54]
             position = list_n[6:9]
             Pposition[atom_count]=position
             list_n[4]=line[21:22]
             list_n[5]=line[22:27].replace(' ','')
             ResinameP[atom_count]=residue+list_n[5]+list_n[4]
             resid_to_name[list_n[5]+list_n[4]] = residue
             resindex=residue+list_n[5]
             #Atomline.append(line)
             Atomline.append(line)
             Pposition[atom_count]=position
   return   resid_to_name,Pposition,ResinameP,Atomline




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
            if out<10 :
                #print (ResinameHL[key1],ResinameHL[key2])
                #print (a,b,out)
                #print (a1)
                residuePair.append([ResinameHL[key1],ResinameE[key2]])
                #print (antiface)
    return   residuePair




range_H='105-117'

range_L='105-117'
range_H_add='56-67'

temH_arr=range_H.split('-')
temH_arr_add=range_H_add.split('-')


allow_mutation=[]
for idx in range(int(temH_arr[0]), int(temH_arr[1])+1):
    allow_mutation.append(str(idx)+"H")

for idx in range(int(temH_arr_add[0]), int(temH_arr_add[1])+1):
    allow_mutation.append(str(idx)+"H")


temL_arr=range_L.split('-')

for idx in range(int(temL_arr[0]), int(temL_arr[1])+1):
    allow_mutation.append(str(idx)+"L")

import sys

filebase=sys.argv[1]


resid_to_name, HLposition, ResinameHL, Atomline=readpdb_allow_m(filebase+'_antibody.pdb',allow_mutation)
Eposition, ResinameE,AtomlineE=readpdb(filebase+'_antigen.pdb')

print ('finished')

residuePair=caculate_pair(HLposition,Eposition,ResinameHL,ResinameE)

print (residuePair)


dict_load=np.load('dict_energy.npy', allow_pickle=True).item()
old_energy={}
real_need_mut=[]

allow_mutation_n=[]
for code1 in allow_mutation:
    for code2 in resid_to_name.keys():
        if  code2[-2].isalpha() and code1==code2[0:-2]+code2[-1]:
            print (code2)
            allow_mutation_n.append(code2)
        elif  not code2[-2].isalpha() and code1==code2:
            allow_mutation_n.append(code2)

print (allow_mutation_n)


for code in allow_mutation_n:
     
     energy=0
     for code_p  in residuePair:
         #print (code,code_p[0][3:])
         if code ==code_p[0][3:]:
             res1=code_p[0][:3]
             res2=code_p[1][:3]
             energy=energy+round(dict_load[res1+'_'+res2],2) 
     if code in resid_to_name.keys():
          print code,resid_to_name[code],energy
          if  energy > 0:
              old_energy[code]=energy
              real_need_mut.append(code)

aa_codes = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']            

print (real_need_mut)

res_arr=[]
for code in real_need_mut:
     #energy=0
     lowest=old_energy[code]
     lowest_res=resid_to_name[code]
     for res1 in  aa_codes:
        energy=0
        for code_p  in residuePair:
            #print (code,code_p[0][3:])
            if code ==code_p[0][3:]:
               #for res1 in  aa_codes:
               #res1=code_p[0][:3]
               #if res1 != resid_to_name[code]:
                     res2=code_p[1][:3]
                     energy=energy+round(dict_load[res1+'_'+res2],2)
        print energy
        if energy<lowest:
            lowest=energy
            lowest_res=res1
     print  code,resid_to_name[code],lowest_res,  lowest
     res_arr.append([resid_to_name[code],lowest_res])

result = list(zip(real_need_mut, res_arr))

print result

import pickle
from collections import OrderedDict

possible_mutations = OrderedDict(result)

with open('possible_mutations.pkl', 'wb') as f:
    pickle.dump(possible_mutations, f)

with open('possible_mutations.pkl', 'rb') as f:
    loaded_mutations = pickle.load(f)

print(loaded_mutations)













