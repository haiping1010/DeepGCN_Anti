
import glob

arr_f=glob.glob("*_poc.pdb")

seq=("ALA","ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")


fw=open("all_nonstardard.txt",'w')

for input_f in arr_f:
#input_f=sys.argv[1]
   fr=open(input_f,"r")
   arr=fr.readlines()
   count=0
   for name in arr:
       if name[17:20] not in seq:
             problem_res=name[17:20]
             print (input_f+" "+str(problem_res))
             fw.write(input_f+" "+str(problem_res))
             fw.write("\n")
             print (input_f)
             break

fw.close()

