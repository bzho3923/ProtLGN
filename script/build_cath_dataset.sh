export k=20

python data.py \
--build_cath \
--protein_dataset data/cath40_k$k \
--c_alpha_max_neighbors $k \
--use_sasa \
--use_bfactor \
--use_dihedral \
--use_coordinate