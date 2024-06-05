# -*- coding: utf-8 -*-
import sys

filebase=sys.argv[1]

aa3to1={
   'ALA':'A', 'VAL':'V', 'PHE':'F', 'PRO':'P', 'MET':'M',
   'ILE':'I', 'LEU':'L', 'ASP':'D', 'GLU':'E', 'LYS':'K',
   'ARG':'R', 'SER':'S', 'THR':'T', 'TYR':'Y', 'HIS':'H',
   'CYS':'C', 'ASN':'N', 'GLN':'Q', 'TRP':'W', 'GLY':'G'
}

aa1to3={value:key  for key,value  in aa3to1.items()}



def readpdb(filename):
   Pposition={}
   ResinameP={}
   Atomline=[]
   index_r=0
   dict_id_map={}
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
         type_n = list_n[2]
         if type_n=='CA':
             index_r=index_r+1
             dict_id_map[str(index_r)]=line[22:27].replace(' ','')
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
   return   Pposition,ResinameP,Atomline,dict_id_map

def save_pdb(pdb_all_line,name):
     fw=open(name,"w")
     for line in pdb_all_line:
         fw.write(line)
     fw.close()

HLposition, ResinameHL, Atomline,dict_id_map=readpdb(filebase+'_antibody.pdb')
Eposition, ResinameE,AtomlineE,dict_id_map_y=readpdb(filebase+'_antigen.pdb')



print dict_id_map

import itertools
from collections import OrderedDict


import pickle
from collections import OrderedDict



with open('mutations_inf.pkl', 'rb') as f:
     mutations_inf = pickle.load(f)

#print(possible_mutations)


'''

possible_mutations = OrderedDict([
    (105, ['ALA', 'GLY']),  # 105位点可能的氨基酸对
    (106, ['THR', 'ASP']),  # 106位点可能的氨基酸对
    # ...
    (112, ['TYR', 'PHE'])  # 112位点可能的氨基酸对
])
'''

#all_combinations = list(itertools.product(*possible_mutations.values()))

'''
for i, combination in enumerate(all_combinations):
    pdb_filename = f'mutation_{i}.pdb'

    with open(pdb_filename, 'w') as pdb_file:
        for position, amino_acids in zip(possible_mutations.keys(), combination):
            pdb_file.write(f'{position}H {amino_acids[0]}\n')
            pdb_file.write(f'{position}L {amino_acids[1]}\n')

    print(f'Saved PDB file: {pdb_filename}')
'''

#print (all_combinations)

#print (possible_mutations.keys())

def  posi_to_res(arr_m):
   position_res={}
   for line in  arr_m:
      arr_tem=line.split('_')
      posi=arr_tem[0].replace('H','')
      pos_T=dict_id_map[posi]
      #combination[i],possible_mutations.keys()[i]
      position_res[pos_T+arr_tem[0][0]] = aa1to3[arr_tem[1]]
   return  position_res


#for i, combination in enumerate( mutations_inf):
for key, value in mutations_inf.items():
         new_Atomline=[]
         #print all_combinations[0][i],possible_mutations.keys()[i]
         position_res=posi_to_res(value)
         #print position_res
         num_m=0
         for line in Atomline:
             if  line[22:27].replace(' ','')+line[21:22] in position_res.keys():
                new_aa=position_res[line[22:27].replace(' ','')+line[21:22]]
                if new_aa !=line[17:20]:
                    if line[11:17].replace(' ','')=="CA":
                        num_m=num_m+1
                    if  line[11:17].replace(' ','')  in ["N","CA","C","O"]:
                        #new_aa=position_res[line[22:26].replace(' ','')+line[21:22]]
                        line_x=line[:17]+ new_aa+line[20:]
                        new_Atomline.append(line_x)
                else:
                    new_Atomline.append(line)
             else:
                 new_Atomline.append(line)

         save_pdb(new_Atomline,"new_antibody_"+key.replace('> ','')+".pdb")

