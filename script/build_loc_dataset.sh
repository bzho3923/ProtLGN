export k=10

python data.py \
    --build_loc \
    --loc_dataset data/location \
    --split train \
    --c_alpha_max_neighbors $k