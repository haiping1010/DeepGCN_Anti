

fr=open('nohup.out','r')

arr=fr.readlines()


print 'epoch:'+'\t'+'AUC:'+'\t'+'TPR:'+'\t'+'precision'+'\t'+'accuracy'+'\t'+'MCC'+'\n'
for line in arr:
    if  line.startswith('epoch:'):
        tem_arr=line.split(':')
        out_line=tem_arr[1].strip()
    if  line.startswith('AUC:'):
        tem_arr=line.split(':')
        out_line=out_line+'\t'+tem_arr[1].strip()
    if  line.startswith('TPR:'):
        tem_arr=line.split(':')
        out_line=out_line+'\t'+tem_arr[1].strip()
    if  line.startswith('precision:'):
        tem_arr=line.split()
        out_line=out_line+'\t'+tem_arr[1].strip()
    if  line.startswith('accuracy:'):
        tem_arr=line.split()
        out_line=out_line+'\t'+tem_arr[1].strip()
    if  line.startswith('MCC:'):
        tem_arr=line.split()
        out_line=out_line+'\t'+tem_arr[1].strip()

        print out_line
