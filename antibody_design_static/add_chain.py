
fr=open('XXXXA_XXXXHL_complex.1_f_cut_redo_run_npt2_out_xx.pdb','r')
arr=fr.readlines()


fw1=open('xxxx_antibody.pdb','w')
fw2=open('xxxx_antigen.pdb','w')


for line in arr:
  if line.startswith('ATOM'):
    tem_arr=line.split()
    if float(tem_arr[1]) <=1833:
         fw1.write( line[0:21]+'H'+line[22:] )
    elif float(tem_arr[1]) >1833  and  float(tem_arr[1])<=3418:
         fw1.write( line[0:21]+'L'+line[22:] )
    elif  float(tem_arr[1])>3418 and float(tem_arr[4])<=6662:
         fw2.write( line[0:21]+'A'+line[22:] )

fw1.close()
fw2.close()


