


frr=open('problem_n.txt','r')

arr_p=frr.readlines()
pro_arr=[]
for linep in arr_p:
    pro_arr.append(linep[:4])


fr=open('temT.txt','r')

fw=open('temT_n.txt','w')


arr_a=fr.readlines()

for line_a in arr_a:
    if line_a[:4] in pro_arr:
        print line_a
    else:
        fw.write(line_a)


fw.close()
fr.close()

fr=open('tem.txt','r')
fw=open('tem_n.txt','w')

arr_a=fr.readlines()

for line_a in arr_a:
    tem_arr=line_a.split()
    if line_a[:4] in pro_arr or tem_arr[1][:4]  in pro_arr:
        print line_a
    else:
        fw.write(line_a)

fw.close()
fr.close()









