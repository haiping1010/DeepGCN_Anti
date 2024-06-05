# -*- coding: utf-8 -*-
import sys

filebase=sys.argv[1]


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
         type = list_n[2]
         #if type == 'CA' and list[3]!= 'UNK':
         if   list_n[3]!= 'UNK' :
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

def save_pdb(pdb_all_line,name):
     fw=open(name,"w")
     for line in pdb_all_line:
         fw.write(line)
     fw.close()

HLposition, ResinameHL, Atomline=readpdb(filebase+'_antibody.pdb')
Eposition, ResinameE,AtomlineE=readpdb(filebase+'_antigen.pdb')





import itertools
from collections import OrderedDict


import pickle
from collections import OrderedDict



with open('possible_mutations.pkl', 'rb') as f:
     possible_mutations = pickle.load(f)

print(possible_mutations)


