 1118  cd ~/work_package/pMD_run_redo_O60488_FACL4
 1119  ls
 1120  cd V001-0135_run
 1121  ls
 1122  vi prod.log 
 1123  cd ../
 1124  cd 
 1125  ls
 1126  cd program/
 1127  ls
 1128  tar -zvxf plumed-2.8.1.tgz
 1129  cd plumed-2.8.1/
 1130  ls
 1131  vi README.md 
 1132  pwd
 1133  mkdir bin
 1134  pwd
 1135  ./configure --prefix=/home/zhanghaiping/program/plumed-2.8.1/bin
 1136  vi README.md 
 1137  make
 1138  make test
 1139  make doc
 1140  ls
 1141  ls bin/
 1142  ls
 1143  ls bin
 1144  cd ../
 1145  ls
 1146  mkdir -p ~/program/gromacs_plumed_2022
 1147  cd ~/program/gromacs_plumed_2022
 1148  ls
 1149  wget https://objects.githubusercontent.com/github-production-release-asset-2e65be/537699/fa59af7c-ceb4-4008-bd0b-c7ed0f438b45?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20231223%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231223T024411Z&X-Amz-Expires=300&X-Amz-Signature=a37242854d5934aac84f369fc5558d3b4b449f1586371c87ade080ae54780d2d&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=537699&response-content-disposition=attachment%3B%20filename%3Dcmake-3.27.9-linux-x86_64.sh&response-content-type=application%2Foctet-stream
 1150  wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.tar.gz
 1151  tar -zvxf cmake-3.27.9-linux-x86_64.tar.gz 
 1152  cd cmake-3.27.9-linux-x86_64/
 1153  ls
 1154  cd bin/
 1155  ls
 1156  ./cmake
 1157  ls
 1158  pwd
 1159  cd work_package
 1160  ls
 1161  cd 
 1162  cd program/
 1163  lls
 1164  ls
 1165  cd plumed-2.8.1/
 1166  ls
 1167  ls bin/
 1168  vi sourceme.sh
 1169  source sourceme.sh
 1170  vi README.md 
 1171  pwd
 1172  ./configure --prefix=/home/zhanghaiping/program/plumed-2.8.1/bin
 1173  make
 1174  ls
 1175  ls python/
 1176  ls bin
 1177  vi README.md 
 1178  umask 022
 1179  make install
 1180  ls
 1181  cd bin/
 1182  ls
 1183  ls bin/
 1184  ls
 1185  cd bin/
 1186  ls
 1187  pwd
 1188  ./plumed
 1189  pwd
 1190  history| grep wget
 1191  cd 
 1192  cd work_package
 1193  ls
 1194  cd pMD_run_redo_O60488_FACL4
 1195  ls
 1196  cd meta_file/
 1197  ls
 1198  vi run_all.bash 
 1199  cd ../
 1200  cd analysis_meta
 1201  ls
 1202  vi run_all.bash 
 1203  bash run_all.bash 
 1204  ls
 1205  bash run_plot.bash 
 1206  ls
 1207  cd ../
 1208  ls
 1209  cd 
 1210  cd program/
 1211  ls
 1212  history | grep wget
 1213  wget https://ftp.gromacs.org/gromacs/gromacs-2022.4.tar.gz
 1214  ls
 1215  tar -zvxf gromacs-2022.4.tar.gz 
 1216  cd gromacs-2022.4/
 1217  ls
 1218  vi README 
 1219  ls
 1220  mkdir build
 1221  cd build/
 1222  ls
 1223  vi ~/readme.txt
 1224  cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=OFF  -DGMX_MPI=on -DGMX_GPU=on -DCMAKE_INSTALL_PREFIX=~/program/gromacs_plumed_2022  -DREGRESSIONTEST_PATH=/home/zhanghaiping/program/gromacs-2019.2/regressiontests--2019.2
 1225  /home/zhanghaiping/program/cmake-3.27.9-linux-x86_64/bin/cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=OFF  -DGMX_MPI=on -DGMX_GPU=on -DCMAKE_INSTALL_PREFIX=~/program/gromacs_plumed_2022  -DREGRESSIONTEST_PATH=/home/zhanghaiping/program/gromacs-2019.2/regressiontests--2019.2
 1226  conda activate py3.8
 1227  /home/zhanghaiping/program/cmake-3.27.9-linux-x86_64/bin/cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=OFF  -DGMX_MPI=on -DGMX_GPU=on -DCMAKE_INSTALL_PREFIX=~/program/gromacs_plumed_2022  -DREGRESSIONTEST_PATH=/home/zhanghaiping/program/gromacs-2019.2/regressiontests--2019.2
 1228  /home/zhanghaiping/program/cmake-3.27.9-linux-x86_64/bin/cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=OFF  -DGMX_MPI=on -DGMX_GPU=CUDA -DCMAKE_INSTALL_PREFIX=~/program/gromacs_plumed_2022  -DREGRESSIONTEST_PATH=/home/zhanghaiping/program/gromacs-2019.2/regressiontests--2019.2 
 1229  mpirun
 1230  which mpirun
 1231  conda deactivate
 1232  which mpirun
 1233  /home/zhanghaiping/program/cmake-3.27.9-linux-x86_64/bin/cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=OFF  -DGMX_MPI=on -DGMX_GPU=CUDA -DCMAKE_INSTALL_PREFIX=~/program/gromacs_plumed_2022  -DREGRESSIONTEST_PATH=/home/zhanghaiping/program/gromacs-2019.2/regressiontests--2019.2 
 1234  conda activate py3.8
 1235  /home/zhanghaiping/program/cmake-3.27.9-linux-x86_64/bin/cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=OFF  -DGMX_MPI=on -DGMX_GPU=CUDA -DCMAKE_INSTALL_PREFIX=~/program/gromacs_plumed_2022  -DREGRESSIONTEST_PATH=/home/zhanghaiping/program/gromacs-2019.2/regressiontests--2019.2 
 1236  ls
 1237  cd ../
 1238  rm -rf build/
 1239  mkdir build
 1240  cd build/
 1241  /home/zhanghaiping/program/cmake-3.27.9-linux-x86_64/bin/cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=OFF  -DGMX_MPI=on -DGMX_GPU=CUDA -DCMAKE_INSTALL_PREFIX=~/program/gromacs_plumed_2022  -DREGRESSIONTEST_PATH=/home/zhanghaiping/program/gromacs-2019.2/regressiontests--2019.2 
 1242  cd ../
 1243  rm -rf build
 1244  conda activate py3.9
 1245  mkdir build
 1246  cd build/
 1247  /home/zhanghaiping/program/cmake-3.27.9-linux-x86_64/bin/cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=OFF  -DGMX_MPI=on -DGMX_GPU=CUDA -DCMAKE_INSTALL_PREFIX=~/program/gromacs_plumed_2022  -DREGRESSIONTEST_PATH=/home/zhanghaiping/program/gromacs-2019.2/regressiontests--2019.2 
 1248  cat /proc/version
 1249  wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda_11.5.0_495.29.05_linux.run
 1250  cd work_package
 1251  ls
 1252  cd pMD_run_redo_O60488_FACL4_pocket2
 1253  ls
 1254  cd meta_file/
 1255  cd ../
 1256  cd analysis_meta
 1257  ls
 1258  ll
 1259  cd ~/work_package/pMD_run_redo_O60488_FACL4_pocket2/analysis_meta
 1260  ls
 1261  pwd
 1262  ls
 1263  cd work_package
 1264  ls
 1265  cd pMD_run_redo_O60488_FACL4_pocket2
 1266  ls
 1267  ls -d *_run
 1268  ls -d *_run > listxx.txt
 1269  vi listxx.txt 
 1270  cd ../
 1271  ls
 1272  cd 
 1273  cd work
 1274  ls
 1275  cd 
 1276  cd work
 1277  cd Schrodinger_screening/
 1278  ls
 1279  cd all_out_select_O60488_FACL4_pocket2
 1280  ls
 1281  cd pocket
 1282  ls
 1283  pwd
 1284  ls
 1285  cd work_package
 1286  ls
 1287  cd 
 1288  cd work
 1289  cd Schrodinger_screening/
 1290  ls
 1291  cd all_out_select_O60488_FACL4
 1292  ls
 1293  cd combine_all/
 1294  ls
 1295  vi output_final.txt 
 1296  ls
 1297  rm all_out_select.sort 
 1298  rm name.list 
 1299  rm output_final.txt 
 1300  ls
 1301  cp -r ../name.list .
 1302  cp -r ~/program/torch/Graph_PDbind_net/all_finished_redo/collect_work3_FACL4_O60488_RG/output_combine.txt .
 1303  vi output_combine.txt 
 1304  ls
 1305  vi output_combine.txt 
 1306  python compare_x.py  name.list  output_combine.txt 
 1307  python compare_x.py  output_combine.txt  name.list 
 1308  vi compare_x.py 
 1309  python compare_x.py  output_combine.txt  name.list 
 1310  ls
 1311  vi name.list 
 1312  ls
 1313  vi name.list 
 1314  cp -r ../complex_n/name_22.list .
 1315  ls
 1316  python compare_x.py  output_combine.txt  name_22.list 
 1317  cd ../
 1318  cd all_out_select_O60488_FACL4_pocket2/
 1319  cd combine_all/
 1320  ls
 1321  rm all_out_select.sort name.list output_final.txt 
 1322  cp -r ../complex_n/name_28.list .
 1323  cp -r ~/program/torch/Graph_PDbind_net/all_finished_redo/collect_work3_FACL4_O60488_pocket2_RG/output_combine.txt .
 1324  ls
 1325  history | grep python
 1326  python compare_x.py  output_combine.txt  name_28.list 
 1327  vi compare_x.py 
 1328  vi output_combine.txt 
 1329  python compare_x.py  output_combine.txt  name_28.list 
 1330  cp -r ~/work/Schrodinger_screening/all_out_select_O60488_FACL4/combine_all/compare_x.py  .
 1331  python compare_x.py  output_combine.txt  name_28.list 
 1332  ks
 1333  ls
 1334  cd work
 1335  ls
 1336  cp -r NAMPT_lig  TMEM43_lig
 1337  cd TMEM43_lig
 1338  ls
 1339  rm nohup.out 
 1340  ls
 1341  rm *.pdb
 1342  ls
 1343  rm *.dat
 1344  ls
 1345  rm *.log
 1346  ls
 1347  rm 7enq_w/ -rf
 1348  ls
 1349  rm protein.*
 1350  ls
 1351  pwd
 1352  ls
 1353  top
 1354  ls
 1355  vi run_all_cofactor.bash 
 1356  rm blast.out 
 1357  ls
 1358  vi run_all_cofactor.bash 
 1359  nohup bash run_all_cofactor.bash &
 1360  top
 1361  ls
 1362  top
 1363  cd 
 1364  ls
 1365  cd program/
 1366  ls
 1367  mkdir antibody_new_data
 1368  cd antibody_new_data/
 1369  ls
 1370  wget https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/archive/all/
 1371  wget https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/archive/all/ -o all_structure.zip
 1372  ll
 1373  unzip all_structure.zip 
 1374  ls
 1375  rm all_structure.zip 
 1376  ls
 1377  rm index.html
 1378  rm index.html.1 
 1379  ls
 1380  pwd
 1381  ls
 1382  pwd
 1383  cd 
 1384  cd work
 1385  ls
 1386  cp  -r TMEM43_lig  RAGE_lig
 1387  cd RAGE_lig
 1388  ls
 1389  rm *.pdb
 1390  ls
 1391  rm nohup.out 
 1392  ls
 1393  rm AF-Q9BTV4-F1-model_v4*
 1394  rm AF-Q9BTV4-F1-model_v4* -rf
 1395  ls
 1396  pwd
 1397  ls
 1398  top
 1399  ls
 1400  vi run_all_cofactor.bash 
 1401  ls
 1402  vi run_all_cofactor.bash 
 1403  nohup bash run_all_cofactor.bash &
 1404  top
 1405  cd work
 1406  ls
 1407  cd 
 1408  cd work_package
 1409  ls -t  *_run/ | head -1
 1410  ls
 1411  rm nohup.out 
 1412  ls
 1413  cd work_package
 1414  ls
 1415  cd MD_complex_li_antibody_full_meta_redo*/
 1416  ls MD_complex_li_antibody_full_meta_redo*/
 1417  ll -t MD_complex_li_antibody_full_meta_redo*/
 1418  ll -t MD_complex_li_antibody_full_meta_redo*/  -h 1
 1419  ll -t MD_complex_li_antibody_full_meta_redo*/  | -h 1
 1420  ll -t MD_complex_li_antibody_full_meta_redo*/  | head  1
 1421  ll -t MD_complex_li_antibody_full_meta_redo*/*  | head  1
 1422  ls -t MD_complex_li_antibody_full_meta_redo*/*  | head  1
 1423  ls -t MD_complex_li_antibody_full_meta_redo*/*  
 1424  ls -t MD_complex_li_antibody_full_meta_redo*/*.bash | head -1  
 1425  ls -t MD_complex_li_antibody_full_meta_redo*/*.bash | head -10
 1426  vi MD_complex_li_antibody_full_meta_redo_four_add/run_all_files.bash
 1427  ls -t MD_complex_li_antibody_full_meta_redo*/*/*.bash | head -10
 1428  vi MD_complex_li_antibody_full_meta_redo_four_add/complex_new_antibody_10007_5_f_cut_run_xxx/run2.bash
 1429  cd MD_complex_li_antibody_full_meta_redo_four_add/complex_new_antibody_10007_5_f_cut_run_xxx
 1430  ls
 1431  vi run2.bash 
 1432  ll --sort=time
 1433  vi run2.bash 
 1434  ll --sort=time
 1435  vi run2.bash
 1436  ls *.pdb
 1437  gmx_mpi pdb2gmx -f  complex_new_antibody_10007_5_f_cut.pdb  -o 1AKI_processed.gro -ff amber99  -ignh  -merge all
 1438  gmx_mpi -h
 1439  gmx_mpi help
 1440  gmx_mpi help ss
 1441  gmx_mpi helpgmx
 1442  vi run2.bash
 1443  gmx_mpi pdb2gmx -f  complex_new_antibody_10007_5_f_cut.pdb  -o 1AKI_processed.gro -ff amber99  -ignh  -merge all -SS
 1444  gmx_mpi pdb2gmx -f  complex_new_antibody_10007_5_f_cut.pdb  -o 1AKI_processed.gro -ff amber99  -ignh  -merge all -ss
 1445  gmx_mpi pdb2gmx -f  complex_new_antibody_10007_5_f_cut.pdb  -o 1AKI_processed.gro -ff amber99  -ignh  -merge all -ss yes
 1446  gmx_mpi pdb2gmx -f  complex_new_antibody_10007_5_f_cut.pdb  -o 1AKI_processed.gro -ff amber99  -ignh  -merge all 
 1447  gmx_mpi pdb2gmx -f  complex_new_antibody_10007_5_f_cut.pdb  -o 1AKI_processed.gro -ff amber99sb  -ignh  -merge all 
 1448  ls
 1449  cd ../
 1450  ls
 1451  cd 
 1452  ls
 1453  cd work
 1454  ls
 1455  cd 
 1456  cd program/
 1457  ls
 1458  cd antibody_new_data/
 1459  ls
 1460  pwd
 1461  ls
 1462  cd 
 1463  cd work_package
 1464  ls
 1465  cd MD_complex_li_antibody_full_meta_redo/
 1466  ls
 1467  cd ../
 1468  ls
 1469  cd MD_complex_li_antibody_full_meta_redo_second
 1470  ls
 1471  cd XXXXA_XXXXHL_complex.1_f_cut_run
 1472  ls
 1473  vi run2.bash 
 1474  pwd
 1475  ls
 1476  cd ../
 1477  ls
 1478  find . -name '#'*
 1479  find . -name '#'* -delete
 1480  ls
 1481  cd XXXXA_XXXXHL_complex.1_f_cut_run
 1482  ls
 1483  pwd
 1484  ls
 1485  pwd
 1486  ls
 1487  cd ../
 1488  ls
 1489  cd ../
 1490  ls
 1491  cd 
 1492  cd program/
 1493  ls
 1494  cd antibody_design_deep3/
 1495  ls
 1496  cd select_prediction/
 1497  ls
 1498  cd ../
 1499  ls
 1500  cd antibody_antigen_design
 1501  ls
 1502  diff design.py  ../antibody_antigen_design-10/design.py 
 1503  vi design.py 
 1504  ls
 1505  cd ../
 1506  ls
 1507  cd antibody_antigen_design-10
 1508  ls
 1509  ls *.py
 1510  vi design.py 
 1511  ls
 1512  ls *.py
 1513  vi design.py 
 1514  cd ../
 1515  ls
 1516  cd ../
 1517  ls
 1518  cd antibody_design_deep_fragment/
 1519  ls
 1520  cd replace_fragement_identify_point/
 1521  ls
 1522  vi design_second_nn.py 
 1523  ll --sort=time *.py
 1524  vi read_mutat.py 
 1525  vi output_file_f.txt 
 1526  ls
 1527  vi design_second_nn.py 
 1528  ls
 1529  cd 
 1530  cd program/
 1531  ls
 1532  cd antibody_new_data/
 1533  ls
 1534  unzip all_structures.zip 
 1535  ls
 1536  ls all_structures
 1537  cd all_structures/
 1538  ls
 1539  cd chothia/
 1540  ls
 1541  cd ../
 1542  ls
 1543  cd chothia
 1544  ls
 1545  vi 7kqb.pdb
 1546  ls
 1547  cd ../
 1548  ls
 1549  cd ../
 1550  ls
 1551  vi extract_anti.py
 1552  vi sabdab_summary_all.tsv 
 1553  ls
 1554  vi extract_anti.py
 1555  vi sabdab_summary_all.tsv 
 1556  vi extract_anti.py
 1557  vi sabdab_summary_all.tsv 
 1558  vi extract_anti.py
 1559  vi sabdab_summary_all.tsv 
 1560  vi extract_anti.py
 1561  vi sabdab_summary_all.tsv 
 1562  vi extract_anti.py
 1563  ls
 1564  vi all_structures/chothia/1aif.pdb 
 1565  vi extract_anti.py
 1566  ls
 1567  cd all_structures/
 1568  ls
 1569  cd raw/
 1570  ls
 1571  cd ../
 1572  ls
 1573  cd ../
 1574  ls
 1575  cp -r all_structures/chothia  .
 1576  ls
 1577  cd chothia/
 1578  cp -r ../extract_anti.py .
 1579  cp -r ../sabdab_summary_all.tsv .
 1580  vi extract_anti.py 
 1581  vi ../sabdab_summary_all.tsv 
 1582  vi extract_anti.py 
 1583  python extract_anti.py 
 1584  vi extract_anti.py 
 1585  python extract_anti.py 
 1586  vi extract_anti.py 
 1587  python extract_anti.py 
 1588  vi extract_anti.py 
 1589  python extract_anti.py 
 1590  ls
 1591  vi 7xy3_antibody.pdb
 1592  vi 7xy3_antigen.pdb
 1593  vi extract_anti.py 
 1594  cp -r 7xy3_*.pdb ~
 1595  ls
 1596  rm *antibody.pdb *_antigen.pdb
 1597  nohup python extract_anti.py &
 1598  top
 1599  ls
 1600  cd program/
 1601  ls
 1602  cd antibody_design_deep_fragment/
 1603  ls
 1604  cd ../
 1605  ls
 1606  cd antibody_design_deep_fragment/
 1607  ls
 1608  cd replace_fragement
 1609  ls
 1610  vi design_second_n.py 
 1611  ls
 1612  cd work8
 1613  ls
 1614  cd usage_zhangting_all_figure_all
 1615  ls
 1616  python combine_n_fig.py 
 1617  ls
 1618  cd work
 1619  ls
 1620  cd TMEM43_lig/
 1621  ls
 1622  top
 1623  ls
 1624  cd AF-Q9BTV4-F1-model_v4/
 1625  ls
 1626  cd cofactor/
 1627  ls
 1628  cd BSITE_AF-Q9BTV4-F1-model_v4/
 1629  ls
 1630  cp -r complex*.pdb ../../
 1631  cd ../../
 1632  ls
 1633  pwd
 1634  cd ../../
 1635  cd RAGE_lig/
 1636  ls
 1637  cd AF-Q15109-F1-model_v4/
 1638  ls
 1639  cd cofactor/
 1640  ls
 1641  cd BSITE_AF-Q15109-F1-model_v4/
 1642  ls
 1643  cp -r complex*.pdb  ../../
 1644  cd ../../
 1645  ls
 1646  pwd
 1647  cd ../
 1648  ls
 1649  cd ../
 1650  ls
 1651  cd 
 1652  cd work_package
 1653  ls
 1654  ls MD_complex_li_antibody_full_meta_redo
 1655  ls MD_complex_li_antibody_*/XXX* -d
 1656  cd MD_complex_li_antibody_full_meta_redo_second/
 1657  ls
 1658  cp -r XXXXA_XXXXHL_complex.1_f_cut_run  XXXXA_XXXXHL_complex.1_f_cut_redo_run
 1659  cd XXXXA_XXXXHL_complex.1_f_cut_redo_run
 1660  ls
 1661  vi gromac2.sh 
 1662  vi run2.bash 
 1663  vi ../run_all_files.bash 
 1664  ls *.pdb
 1665  bash run2.bash XXXXA_XXXXHL_complex.1_f_cut
 1666  ls
 1667  vi topol.top 
 1668  rm prod.xtc
 1669  rm prod.gro
 1670  nohup bash gromac2.sh &
 1671  vi gromac2.sh 
 1672  top
 1673  nvidia-smi
 1674  ls
 1675  top
 1676  cd work8
 1677  ls
 1678  cd usage_zhangting_all_figure_all
 1679  ls
 1680  rm table_*
 1681  ls
 1682  rm nohup.out 
 1683  nohup bash run_all_n.bash &
 1684  ls
 1685  cd work
 1686  ls
 1687  top
 1688  ls
 1689  cd ../
 1690  cd work_package
 1691  ls
 1692  cd ls
 1693  cd database
 1694  ls
 1695  cd ~/work_package/database
 1696  ls
 1697  nohup cp -r work3_VS_n_O60488_FACL4   work3_VS_n_O60488_FACL4_template &
 1698  ls
 1699  cd 
 1700  cd program/
 1701  ls
 1702  cd torch
 1703  ls
 1704  cd Graph_PDbind_net/
 1705  ls
 1706  cd all_finished_redo
 1707  ls
 1708  nohup cp -r work3_VS_n_general_FACL4_O60488_RG_cutoff0.8  work3_VS_n_general_FACL4_O60488_RG_cutoff0.8_template &
 1709  ls
 1710  cd work3_VS_n_general_FACL4_O60488_RG_cutoff0.8_template
 1711  ls
 1712  rm data1/ -rf
 1713  ls
 1714  rm *.log
 1715  ls
 1716  rm output_*.txt
 1717  ls
 1718  rm nohup.out 
 1719  ls
 1720  du -h
 1721  cd all_file/
 1722  ls
 1723  cd ../
 1724  ls
 1725  cd ../
 1726  ls
 1727  for i in {1..4}; do cp -r  work3_VS_n_general_FACL4_O60488_RG_cutoff0.8_template  work3_VS_n_general_Q9BTV4_TMEM43_RG_cutoff0.8_$i; echo $i  ; done
 1728  ls
 1729  cp -r work3_VS_n_general_FACL4_O60488_RG_cutoff0.8_template  work3_VS_n_general_RAGE_RG_cutoff0.8
 1730  cd bcup_baodei/
 1731  ls
 1732  cd antibody_antigen_all/
 1733  ls
 1734  vi collect_inf.py 
 1735  ls *.py
 1736  vi obtain_list.
 1737  vi obtain_list.py 
 1738  ls
 1739  cd zdock_redo_new
 1740  ls
 1741  vi tem.txt 
 1742  grep tem *.py
 1743  vi random_cp2.py
 1744  ls
 1745  cd ../
 1746  ls
 1747  cd ~/bcup_baodei/antibody_antigen_all/positive_final_n
 1748  ls
 1749  cd work_package
 1750  ls
 1751  cd database
 1752  ls
 1753  cd work3_VS_n_O60488_FACL4_template
 1754  ls
 1755  rm *.log
 1756  ls
 1757  ls mp_data_??
 1758  rm -rf mp_data_??/p*
 1759  rm -rf mp_data_??/s*
 1760  ls mp_data_??
 1761  ls
 1762  rm all_out*
 1763  ls
 1764  rm nohup.out 
 1765  du -h
 1766  ls
 1767  cd ../
 1768  ls
 1769  history | grep cp
 1770  ls
 1771  for i in {1..4}; do cp -r  work3_VS_n_O60488_FACL4_template   work3_VS_n_Q9BTV4_TMEM43_$i; echo $i  ; done
 1772  ls
 1773  cd program/
 1774  ls
 1775  cd antibody_new_data/
 1776  ls
 1777  cd chothia/
 1778  ls
 1779  vi extract_anti.py 
 1780  ls
 1781  cd ../
 1782  ls
 1783  pwd
 1784  cd bcup_baodei/
 1785  ls
 1786  cd  antibody_antigen_all/
 1787  ls
 1788  cd positive_final_n
 1789  ls
 1790  cd docking_complex/
 1791  ls
 1792  cd ../
 1793  ls
 1794  grep block *.py
 1795  grep block *.bash
 1796  vi run_block.bash 
 1797  ls
 1798  vi  python_2_pp_d.py
 1799  vi python_2_pp.py
 1800  ls
 1801  ls
 1802  vi python_block_n.py 
 1803  cd ../
 1804  ls
 1805  find . -name python_block*
 1806  cd download_nn
 1807  ls
 1808  vi run_block.bash 
 1809  cd ../
 1810  find . -name inf_index.txt
 1811  ls
 1812  cd positive_final
 1813  ls
 1814  diff run_block_n.bash run_block.bash 
 1815  vi run_block_n.bash 
 1816  vi all_chain_sum.txt
 1817  ls
 1818  cp -r run_block_n.bash  /home/zhanghaiping/program/antibody_new_data/chothia/
 1819  vi run_block_n.bash 
 1820  cp -r all_chain_sum.txt  /home/zhanghaiping/program/antibody_new_data/chothia/
 1821  cp -r  python_block_n.py  /home/zhanghaiping/program/antibody_new_data/chothia/
 1822  cd ~/program/antibody_new_data
 1823  ls
 1824  nohup cp  -r  ~/bcup_baodei/antibody_antigen_all/zdock_redo_new/*  . &
 1825  ls
 1826  cat nohup.out 
 1827  ls
 1828  ls zdock_redo_new/
 1829  ls
 1830  top
 1831  ls
 1832  cd chothia/
 1833  ls
 1834  mkdir old
 1835  mv *_antibody.pdb old
 1836  ls
 1837  mv *_antigen.pdb old
 1838  ls
 1839  pwd
 1840  ls
 1841  vi extract_anti.py 
 1842  cp -r extract_anti.py  extract_inf.py
 1843  vi extract_inf.py 
 1844  vi all_chain_sum.txt 
 1845  vi extract_inf.py 
 1846  python extract_inf.py 
 1847  ls
 1848  vi all_chain_sum.txt 
 1849  ls
 1850  rm -rf nohup.out  old
 1851  vi all_chain_sum.txt 
 1852  grep 4fql  sabdab_summary_all.tsv 
 1853  vi sabdab_summary_all.tsv 
 1854  vi extract_inf.py 
 1855  python extract_inf.py 
 1856  ls
 1857  vi all_chain_sum.txt 
 1858  vi extract_inf.py 
 1859  python extract_inf.py 
 1860  vi all_chain_sum.txt 
 1861  ls *.pdb |wc
 1862  vi all_chain_sum.txt 
 1863  awk -F '.'  '{print $1}'  all_chain_sum.txt 
 1864  awk -F '.'  '{print $1}'  all_chain_sum.txt | sort | uniq
 1865  awk -F '.'  '{print $1}'  all_chain_sum.txt | sort | uniq |wc
 1866  vi all_chain_sum.txt 
 1867  sort all_chain_sum.txt  
 1868  ls
 1869  vi all_chain_sum.txt 
 1870  awk -F '.'  '{print $1}'  all_chain_sum.txt | sort | uniq |wc
 1871  awk -F '.'  '{print $1}'  all_chain_sum.txt | sort | uniq > uniq_list.txt
 1872  vi uniq_list.txt 
 1873  cp -r ~/compare.py .
 1874  vi compare.py 
 1875  python compare.py  uniq_list.txt   all_chain_sum.txt 
 1876  vi compare.py 
 1877  python compare.py  uniq_list.txt   all_chain_sum.txt |wc
 1878  python compare.py  uniq_list.txt   all_chain_sum.txt > all_chain_sum_uniq.txt 
 1879  vi all_chain_sum_uniq.txt 
 1880  ls
 1881  vi run_block_n.bash 
 1882  nohup bash run_block_n.bash &
 1883  top
 1884  ls
 1885  cat nohup.out 
 1886  vi run_block_n.bash 
 1887  rm nohup.out 
 1888  nohup bash run_block_n.bash &
 1889  ls
 1890  top
 1891  ls
 1892  top
 1893  ls
 1894  ls *_antibody.pdb
 1895  vi 1cft.pdbBA_antibody.pdb
 1896  vi 1cft.pd*
 1897  ls 1cft.pd*
 1898  vi ls *_antigen.pdb
 1899  ls *_antibody.pdb
 1900  ls
 1901  vi all_chain_sum_uniq.txt 
 1902  ls
 1903  top
 1904  pkill python
 1905  ls
 1906  rm *_antibody.pdb
 1907  ls
 1908  rm *_antigen.pdb
 1909  ls
 1910  vi all_chain_sum_uniq.txt 
 1911  ls
 1912  rm -rf antibody antigen
 1913  ls
 1914  rm nohup.out 
 1915  ls
 1916  top
 1917  pkill bash
 1918  pkill python
 1919  ls
 1920  rm *antigen.pdb
 1921  rm *antibody.pdb
 1922  ls
 1923  vi run_block_n.bash 
 1924  nohup bash run_block_n.bash &
 1925  ls
 1926  ls *antigen.pdb
 1927  vi all_chain_sum_uniq.txt 
 1928  grep 1cfv  sabdab_summary_all.tsv
 1929  vi sabdab_summary_all.tsv
 1930  ls
 1931  vi all_chain_sum_uniq.txt 
 1932  grep 1mcc  all_chain_sum_uniq.txt 
 1933  grep 1mcc  sabdab_summary_all.tsv 
 1934  vi extract_inf.py 
 1935  python extract_inf.py 
 1936  vi extract_inf.py 
 1937  python extract_inf.py 
 1938  ls
 1939  vi all_chain_sum.txt 
 1940  history| grep awk
 1941  awk -F '.'  '{print $1}'  all_chain_sum.txt | sort | uniq > uniq_list.txt
 1942  python compare.py  uniq_list.txt  all_chain_sum.txt 
 1943  python compare.py  uniq_list.txt  all_chain_sum.txt > all_chain_sum_uniq.txt 
 1944  vi all_chain_sum_uniq.txt 
 1945  vi uniq_list.txt 
 1946  awk -F ':'  '{print $1}'  all_chain_sum.txt | sort | uniq > uniq_list.txt
 1947  python compare.py  uniq_list.txt  all_chain_sum.txt > all_chain_sum_uniq.txt 
 1948  ls
 1949  top
 1950  pkill bash
 1951  pkill python
 1952  ls
 1953  rm *_antigen.pdb *_antibody.pdb
 1954  ls
 1955  wc all_chain_sum_uniq.txt 
 1956  vi all_chain_sum_uniq.txt 
 1957  ls
 1958  vi nohup.out 
 1959  vi all_chain_sum_uniq.txt 
 1960  ls
 1961  nohup bash run_block_n.bash &
 1962  top
 1963  ls
 1964  pkill bash
 1965  ls
 1966  ls *antibody.pdb
 1967  ls *antibody.pdb  *antigen.pdb
 1968  rm  *antibody.pdb  *antigen.pdb
 1969  ls
 1970  rm nohup.out 
 1971  ls
 1972  vi run_block_n.bash 
 1973  vi python_block_n.py 
 1974  nohup bash run_block_n.bash &
 1975  top
 1976  cd 
 1977  ls
 1978  cd  potential2_Anti
 1979  ls
 1980  ls ../potential2_Anti_cutoff1
 1981  ls
 1982  rm "'"
 1983  ls
 1984  rm nohup.out 
 1985  ls
 1986  cd tem
 1987  ls
 1988  vi python_2_anti_stat_neighbor.py 
 1989  ls
 1990  vi summary.txt 
 1991  ls *.pdb |wc
 1992  vi python_2_anti_stat_neighbor.py 
 1993  cd ../
 1994  ls
 1995  cd potential2_Anti
 1996  ls
 1997  display heatplot.png 
 1998  cd 
 1999  ls
 2000  cd program/
 2001  ls
 2002  cd antibody_new_data/
 2003  ls
 2004  cd zdock_redo_new/
 2005  ls
 2006  cd ../
 2007  ls
 2008  mkdir xxxx
 2009  cd xxxx/
 2010  ls
 2011  mv ../docking*.bash .
 2012  ls
 2013  ls ../
 2014  mv ../tem-*.txt .
 2015  ls
 2016  cd ../
 2017  ls
 2018  cd chothia/
 2019  ls
 2020  ls antibody/
 2021  ls antibody/*.pdb |wc
 2022  ls antigen/*.pdb |wc
 2023  ls
 2024  top
 2025  ls
 2026  vi python_block_n.py 
 2027  ls *_antigen.pdb |wc
 2028  ls *_antibody.pdb |wc
 2029  ls
 2030  vi all_chain_sum_uniq.txt 
 2031  vi run_block_n.bash 
 2032  vi python_block_n.py 
 2033  for name in *_antigen.pdb; do base=${name:0:4}; echo $base; done
 2034  for name in *_antigen.pdb; do base=${name:0:4}; echo $base; mv $name  $base'_antigen.pdb';done
 2035  for name in *_antibody.pdb; do base=${name:0:4}; echo $base; mv $name  $base'_antibody.pdb';done
 2036  ls
 2037  rm *_antibody.pdb *.antigen.pdb
 2038  rm -rf *_antibody.pdb *.antigen.pdb
 2039  rm -rf *_antibody.pdb *_antigen.pdb
 2040  ls
 2041  rm nohup.out 
 2042  vi all_chain_sum_uniq.txt 
 2043  nohup bash run_block_n.bash &
 2044  ls
 2045  ls *_antigen.pdb
 2046  top
 2047  sl
 2048  ls
 2049  ls *_antigen.pdb
 2050  ls
 2051  cd 
 2052  ls
 2053  nohup cp -r potential2_Anti  potential2_Anti_new &
 2054  ls
 2055  cd potential2_Anti_vec
 2056  ls
 2057  cd vec_to_acid/
 2058  ls
 2059  cd ../
 2060  ls
 2061  cd ../
 2062  ls
 2063  nohup cp -r potential2_Anti_vec  potential2_Anti_new_vec &
 2064  ls
 2065  cd program/
 2066  ls
 2067  cd antibody_design_deep_fragment/
 2068  ls
 2069  cd ../
 2070  ls
 2071  nohup cp -r antibody_design_deep_fragment  antibody_design_deep_fragment2 &
 2072  top
 2073  ls
 2074  cd antibody_design_deep_fragment2
 2075  ls
 2076  vi all_CDR.txt 
 2077  vi HCDR3.fasta 
 2078  vi convert_fasta.py 
 2079  vi out_seq_f.txt
 2080  ls *.bash
 2081  vi all_CDR.txt 
 2082  grep H3 *
 2083  vi out_seq_f.txt 
 2084  ls
 2085  vi HCDR3.fasta 
 2086  vi all_CDR.txt 
 2087  vi LCDR3_uniq.fasta
 2088  ls
 2089  cd ../
 2090  cd antibody_design_deep_fragment
 2091  ls
 2092  vi nohup.out 
 2093  cd ../
 2094  ls
 2095  cd antibody_design_deep_fragment2
 2096  ls
 2097  rm nohup.out 
 2098  vi all_CDR.txt 
 2099  grep ":" all_CDR.txt > all_CDR_seq.txt
 2100  vi all_CDR_seq.txt 
 2101  grep "??:" all_CDR.txt > all_CDR_seq.txt
 2102  vi all_CDR_seq.txt 
 2103  grep -E ".{2}:" all_CDR.txt > all_CDR_seq.txt
 2104  vi all_CDR_seq.txt 
 2105  grep -E ".{2}:" all_CDR.txt > all_CDR_seq.txt
 2106  vi all_CDR_seq.txt 
 2107  grep -E "^.{2}:" all_CDR.txt > all_CDR_seq.txt
 2108*  all_CDR_seq.txt 
 2109  vi out_seq_f.fasta
 2110  awk -F ' ' all_CDR_seq.txt '{print $1}'
 2111  awk -F ' '  '{print $1}'  all_CDR_seq.txt 
 2112  awk -F ' '  '{print $2}'  all_CDR_seq.txt 
 2113  awk -F ' '  '{print $2}'  all_CDR_seq.txt | sort | uniq
 2114  awk -F ' '  '{print $2}'  all_CDR_seq.txt | sort | uniq > all_CDR_seq_uniq.txt 
 2115  ls
 2116  history
 2117  history > readme.txt



makeblastdb -in  all_CDR_seq_uniq.fasta -dbtype prot
blastp -query your_query_sequence.fasta -db all_CDR_seq_uniq.fasta  -out  xxx 
 blastp -query your_query_sequence.fasta -db all_CDR_seq_uniq.fasta  -out  xxx  -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"














