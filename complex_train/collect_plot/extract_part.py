
import sys
input_f=sys.argv[1]

fr=open(input_f,'r')
arr=fr.readlines()
index=0
for line in arr:
    if index ==0:
        print line.strip()
    if index !=0 and line !='\n':
      arr_tem=line.split('\t')
      #print arr_tem[0]
      if int(arr_tem[0])%100==0:
          print  line.strip()

    index=index+1
