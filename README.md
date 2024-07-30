# DeepGCN_Anti
The antibody_design_static folder contains static based potential based mutation screening pipeline. 
1. The file run_all_query.bash was used to do the sequence blast over the CDR dataset, and obtain the convolutionary constraint.
2. The read_mutat.py in replace_fragement_identify_point_?? folder was used to extract the convolutionary constraint(collect information which point can mutate, and mutate to which type, save those information into possible_mutations.pkl), for user own antibody, the CDR region defination in the read_mutat.py modify accordingly.
3. The folder antibody_design_static/all_generate_single_m is used to generate the all mutated antibody-antigen complex structures based on the evolutionary restrain, this antibody_design_static/all_generate_single_m/design_second_nn.py is the major script to achieve this.
4. antibody_design_static/static_eval  this folder is use constructed static potential do the screening with 6Å as cutoff for obtaining the interface residue pairs, and only those added up parwised static potential have relative low value will keep. antibody_design_static/static_eval  this folder is use constructed static potential do the screening with 6Å as cutoff for obtaining the interface residue pairs, and only those added up parwised static potential have relative low value will keep. Similarly, static_eval_cutoff10 folder is use constructed static potential do the screening with 10Å as cutoff for obtaining the interface residue pairs.

 




The complex_train contains the code and script for training the DeepGCN_anti.


xxx folder contains the script for MD simulation and metadynamics simulation

xxx folder contains the script to construct the static potential.
