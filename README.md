# DeepGCN_Anti
The antibody_design_static folder contains static based potential based mutation screening pipeline. 
1. The file run_all_query.bash was used to do the sequence blast over the CDR dataset, and obtain the convolutionary constraint.
2. The read_mutat.py in replace_fragement_identify_point_?? folder was used to extract the convolutionary constraint(collect information which point can mutate, and mutate to which type, save those information into possible_mutations.pkl), for user own antibody, the CDR region defination in the read_mutat.py modify accordingly.
3. 

 




The complex_train contains the code and script for training the DeepGCN_anti.


xxx folder contains the script for MD simulation and metadynamics simulation

xxx folder contains the script to construct the static potential.
