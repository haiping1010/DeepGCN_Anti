

fr=open('temT.txt','r')

arr=fr.readlines()

count=0
out_count=0
arr_contain=[]
for name in arr:
    count=count+1
    arr_contain.append(name)
    if count%500==0:
        out_count=out_count+1
        fw=open('temT_'+str(out_count)+'.txt','w')
        for namex in arr_contain:
            fw.write(namex)
        fw.close()
        arr_contain=[]

out_count=out_count+1
fw=open('temT_'+str(out_count)+'.txt','w')

for namex in arr_contain:
            fw.write(namex)

fw.close()


    

