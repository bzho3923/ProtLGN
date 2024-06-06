
CUDA_VISIBLE_DEVICES=0 python mutant_predict.py \
    --checkpoint ckpt/ProtLGN.pt \
    --c_alpha_max_neighbors 10 \
    --gnn egnn \
    --use_sasa \
    --layer_num 6 \
    --gnn_config src/Egnnconfig/egnn_mutant.yaml \
    --mutant_dataset data/example