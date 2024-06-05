for name  in   *_dic_contact_vec_e.npy
do

	base=${name%_dic_contact_vec_e.npy}
	nohup  python    deep_dense_FC_n_run_redo_load.py   $name  >  $base'_pred.log'  2>&1&

done
