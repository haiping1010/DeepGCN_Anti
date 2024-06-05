
begin=96

def caculate_mut(frag_ori,frag_rep,beg_num,dict_p_r):
     arr_n=[]
     for i in  range(len(frag_ori)):
         if frag_ori[i] != frag_rep[i]:
               arr_n.append('H'+str(i+int(beg_num)+begin)+'_'+frag_rep[i])
               dict_p_r['H'+str(i+int(beg_num)+begin)].append(frag_rep[i])

     return  arr_n,dict_p_r


    

fr_f=open('output_file_f.txt','r')

arr_f=fr_f.readlines()

dict_m={}

dict_p_r={}
ori_seq='ATGNADCSGGSCYNWFDP'
for i in range(len(ori_seq)):
         dict_p_r['H'+str(i+1+begin)]=[]


for line in arr_f:
    if  line.startswith('>'):
        name=line[1:].strip()
    elif line.startswith('Query'):
        arr_tem=line.split()
        #arr_tem[1]
    elif  line.startswith('Sbjct'):
         
         arr_tem2=line.split()
         if '-' not in arr_tem[2] and '-' not in arr_tem2[2]:
               dict_m[name],dict_p_r=caculate_mut(arr_tem[2],arr_tem2[2],arr_tem[1],dict_p_r)


import pickle


with open('mutations_inf.pkl', 'wb') as f:
    pickle.dump(dict_m, f)

with open('mutations_inf.pkl', 'rb') as f:
    mutations_inf = pickle.load(f)

print(mutations_inf)

print dict_p_r

def count_acid(acid, value):
   count=0
   for code in value:
       if code==acid:
           count=count+1
   return count
dict_p_r_n={}
for key, value in  dict_p_r.items():
    print(key, value)
    value_set=list(set(value))
    dict_p_r_n[key]=[]
    for acid in value_set:
        if count_acid(acid,value) >=1:
            dict_p_r_n[key].append(acid)

print dict_p_r_n

         
dict_without_empty = {k: v for k, v in dict_p_r_n.items() if v}


print dict_without_empty

def readpdb(filename):
   Pposition={}
   ResinameP={}
   Atomline=[]
   index_r=0
   dict_id_map={}
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

             resid_to_name[list_n[5]+list_n[4]] = residue

             ResinameP[atom_count]=residue+list_n[5]+list_n[4]
             resindex=residue+list_n[5]
             #Atomline.append(line)
             Atomline.append(line)
             Pposition[atom_count]=position
   return   resid_to_name,Pposition,ResinameP,Atomline,dict_id_map

filebase='xxxx'
resid_to_name, HLposition, ResinameHL, Atomline,dict_id_map=readpdb(filebase+'_antibody.pdb')


aa3to1={
   'ALA':'A', 'VAL':'V', 'PHE':'F', 'PRO':'P', 'MET':'M',
   'ILE':'I', 'LEU':'L', 'ASP':'D', 'GLU':'E', 'LYS':'K',
   'ARG':'R', 'SER':'S', 'THR':'T', 'TYR':'Y', 'HIS':'H',
   'CYS':'C', 'ASN':'N', 'GLN':'Q', 'TRP':'W', 'GLY':'G'
}

aa1to3={value:key  for key,value  in aa3to1.items()}

dict_final={}
for key, value in  dict_without_empty.items():
      print(key, value)
      
      posi=key.replace('H','')
      pos_T=dict_id_map[posi]
      #combination[i],possible_mutations.keys()[i]
      value_n=[aa1to3[x] for x in value]
      print (value_n)
      value_n.append(resid_to_name[pos_T+'H'])
      dict_final[pos_T+'H'] = value_n

print dict_final


import pickle

with open('possible_mutations.pkl', 'wb') as f:
    pickle.dump(dict_final, f)

with open('possible_mutations.pkl', 'rb') as f:
    mutations_inf = pickle.load(f)

