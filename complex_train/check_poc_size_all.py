
import sys
import glob

all_files_arr=glob.glob("*_poc.pdb")
fw=open("all_problem_small10.txt",'w')
fww=open("all_pock_siz.txt",'w')

for input_f in all_files_arr:
#input_f=sys.argv[1]
   fr=open(input_f,"r")
   arr=fr.readlines()
   count=0
   for name in arr:
       if name[13:15] == 'CA':
             count=count+1
   print (count)
   fww.write(input_f+" "+str(count))
   fww.write("\n")
   if count<=10:
       fw.write(input_f)
       fw.write("\n")
fw.close()
fww.close()
