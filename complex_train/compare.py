import os
import os
import shutil
if (len(os.sys.argv)<2):

    print ("usage: python compare.py file1 file2")


nf=open(os.sys.argv[1], 'r')
old=[]
oldvalue=[]
oldline=[]

print (os.sys.argv[1])
newlines=nf.readlines()

f=open(os.sys.argv[2], 'r')

lines=f.readlines()
for newname in newlines:
    index=0
    for name in lines:
         if newname[0:6].upper() == name[0:6].upper():
              index=index+1

              print (newname.strip())
              break
