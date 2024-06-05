import shutil, random, os
#filenames = random.sample(os.listdir(dirpath), 4000)
import glob
aa=glob.glob("*_antigen.pdb")
##filenames = random.sample(os.listdir(dirpath), 3)
#filenames = random.sample(aa, 3)

bb=glob.glob("*_antibody.pdb")



fw=open("tem.txt", "w")
print (fw.name)

for pname in aa:
    for lname in bb:
        if pname[0:4] ==lname[0:4]:
            fw.write(pname+" "+ lname+"\n")

fw.close()
