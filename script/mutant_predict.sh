GNN_TYPE=egnn

CUDA_VISIBLE_DEVICES=7 python mutant_predict.py \
    --checkpoint result/weight/Sep26_loty.pt \
    --c_alpha_max_neighbors 10 \
    --gnn $GNN_TYPE \
    --use_sasa \
    --layer_num 6 \
    --gnn_config src/Egnnconfig/$GNN_TYPE.yaml \
    --mutant_dataset data/proteingym