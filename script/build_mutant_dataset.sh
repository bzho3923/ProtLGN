export k=10

python data.py \
--build_mutant \
--mutant_dataset data/proteingym_valid \
--c_alpha_max_neighbors $k