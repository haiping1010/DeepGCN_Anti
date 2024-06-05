
import sys

input_f=sys.argv[1]

fr=open(input_f,"r")
arr=fr.readlines()

count=0
for name in arr:
    if name[13:15] == 'CA':
        count=count+1

print (count)
