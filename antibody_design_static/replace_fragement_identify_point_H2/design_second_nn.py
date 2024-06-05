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


'''

possible_mutations = OrderedDict([
    (105, ['ALA', 'GLY']),  # 105位点可能的氨基酸对
    (106, ['THR', 'ASP']),  # 106位点可能的氨基酸对
    # ...
    (112, ['TYR', 'PHE'])  # 112位点可能的氨基酸对
])
'''

all_combinations = list(itertools.product(*possible_mutations.values()))

'''
for i, combination in enumerate(all_combinations):
    pdb_filename = f'mutation_{i}.pdb'

    with open(pdb_filename, 'w') as pdb_file:
        for position, amino_acids in zip(possible_mutations.keys(), combination):
            pdb_file.write(f'{position}H {amino_acids[0]}\n')
            pdb_file.write(f'{position}L {amino_acids[1]}\n')

    print(f'Saved PDB file: {pdb_filename}')
'''

print (all_combinations)

print (possible_mutations.keys())

def  posi_to_res(combination):
   position_res={}
   for i in range(len(combination)):
      #combination[i],possible_mutations.keys()[i]
      position_res[possible_mutations.keys()[i]] = combination[i]
   return  position_res


for i, combination in enumerate(all_combinations):

         new_Atomline=[]
         #print all_combinations[0][i],possible_mutations.keys()[i]
         position_res=posi_to_res(combination)
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

         save_pdb(new_Atomline,"new_antibody_"+str(i)+"_"+str(num_m)+".pdb")

'''
# 遍历所有组合并保存为PDB文件
for i, combination in enumerate(all_combinations):
    pdb_filename = f'mutation_{i}.pdb'

    # 根据需要创建PDB文件，并将组合写入文件
    with open(pdb_filename, 'w', encoding='utf-8') as pdb_file:
        for (position, _), amino_acids in zip(possible_mutations.items(), combination):
            pdb_file.write(f'{position}H {amino_acids[0]}\n')
            pdb_file.write(f'{position}L {amino_acids[1]}\n')

    print(f'Saved PDB file: {pdb_filename}')

'''
