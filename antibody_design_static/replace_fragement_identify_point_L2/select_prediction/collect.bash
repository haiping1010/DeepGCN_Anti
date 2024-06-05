
mkdir collect

cat    all_out_select.sort  | while read line
do

IFS=' ' read -r -a array <<< $line
##wget 'http://zinc15.docking.org/substances/'${array[0]}'.sdf'

#cp -r  /home/zhanghaiping/work/database3/sdf_all_L1_L4_L6/*/${array[0]}'.sdf'  .


#echo ${array[0]:17}
cp  -r ../${array[0]}/${array[1]}   collect/${array[1]}



done


