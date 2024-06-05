
import glob

all_files=glob.glob('*_f.pdb')

#1-45  A

#259-316  A
arr_resid=[]
for index in range(1,46):
     arr_resid.append(str(index))

for index in   range(259,317):
     arr_resid.append(str(index))

for name in all_files:
    fr=open(name,'r')
    fw=open(name.replace('_f.pdb','_f_cut.pdb'),'w')
    arr_all=fr.readlines()
    for line in arr_all:
        if  line[22:26].replace(' ','') in arr_resid  and line[21:22] =='A':
            print ("cut"+line)
        else:
            fw.write(line)
    fw.close()
    fr.close()

    


